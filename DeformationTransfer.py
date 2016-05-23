#Code by Chris Tralie, Parit Burintrathikul, Justin Wang, Lydia Xu, Billy Wan, and Jay Wang
import sys
sys.path.append("S3DGLPy")
from Primitives3D import *
from PolyMesh import *
import numpy as np
from scipy import sparse
import scipy.io as sio
from scipy.linalg import norm
from scipy.sparse.linalg import lsqr

def loadBaselKeypointMesh():
    (VPos, VColors, ITris) = loadOffFileExternal("BUMesh.off")    
    return (VPos, ITris)

class VideoMesh(object):
    def __init__(self):
        self.Frames = np.array([])
        self.ITris = np.array([])

    #Initialize the basel video with the first (neutral) frame filled in
    #and the rest blank
    def initBaselVideo(self, filename, NFrames):
        (a, b, self.ITris) = loadOffFileExternal("BUMesh.off")
        #Grab the keypoints of the chosen basel model
        shape = sio.loadmat(filename)['shape']
        shape = np.reshape(shape, [len(shape)/3, 3])
        idx = sio.loadmat("BaselBUKeypointsIdx")['idx']-1
        idx = idx.flatten()
        shape = shape[idx, :]
        self.Frames = np.zeros((NFrames, shape.shape[0], shape.shape[1]))
        self.Frames[0, :, :] = shape
    
    #Load in a bunch of bnd files, assuming the first one is a neutral
    #expression
    def initBUVideo(self, paths):
        (a, b, self.ITris) = loadOffFileExternal("BUMesh.off")
        X1 = np.loadtxt(paths[0])
        X1 = X1[:, 1::]
        NFrames = len(paths)
        self.Frames = np.zeros((NFrames, X1.shape[0], X1.shape[1]))
        self.Frames[0, :, :] = X1
        for i in range(1, NFrames):
            X = np.loadtxt(paths[i])
            X = X[:, 1::]
            self.Frames[i, :, :] = X
    
    def saveFramesOff(self, prefix):
        for i in range(self.Frames.shape[0]):
            VPos = self.Frames[i, :, :]
            fout = open("%s%i.off"%(prefix, i), "w")
            fout.write("OFF\n%i %i 0\n"%(VPos.shape[0], self.ITris.shape[0]))
            for i in range(VPos.shape[0]):
                fout.write("%g %g %g\n"%(VPos[i, 0], VPos[i, 1], VPos[i, 2]))
            for i in range(self.ITris.shape[0]):
                fout.write("3 %g %g %g\n"%(self.ITris[i, 0], self.ITris[i, 1], self.ITris[i, 2]))
            fout.close()
        

def getLaplacianMatrixCotangent(mesh, anchorsIdx, anchorWeights = 1):
    VPos = mesh.VPos
    ITris = mesh.ITris
    N = VPos.shape[0]
    M = ITris.shape[0]
    #Allocate space for the sparse array storage, with 2 entries for every
    #edge for every triangle (6 entries per triangle); one entry for directed 
    #edge ij and ji.  Note that this means that edges with two incident triangles
    #will have two entries per directed edge, but sparse array will sum them 
    I = np.zeros(M*6)
    J = np.zeros(M*6)
    V = np.zeros(M*6)
    
    #Keep track of areas of incident triangles and the number of incident triangles
    IA = np.zeros(M*3)
    VA = np.zeros(M*3) #Incident areas
    VC = 1.0*np.ones(M*3) #Number of incident triangles
    
    #Step 1: Compute cotangent weights
    for shift in range(3): 
        #For all 3 shifts of the roles of triangle vertices
        #to compute different cotangent weights
        [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
        dV1 = VPos[ITris[:, i], :] - VPos[ITris[:, k], :]
        dV2 = VPos[ITris[:, j], :] - VPos[ITris[:, k], :]
        Normal = np.cross(dV1, dV2)
        #Cotangent is dot product / mag cross product
        NMag = np.sqrt(np.sum(Normal**2, 1))
        cotAlpha = np.sum(dV1*dV2, 1)/NMag
        I[shift*M*2:shift*M*2+M] = ITris[:, i]
        J[shift*M*2:shift*M*2+M] = ITris[:, j] 
        V[shift*M*2:shift*M*2+M] = cotAlpha
        I[shift*M*2+M:shift*M*2+2*M] = ITris[:, j]
        J[shift*M*2+M:shift*M*2+2*M] = ITris[:, i] 
        V[shift*M*2+M:shift*M*2+2*M] = cotAlpha
        if shift == 0:
            #Compute contribution of this triangle to each of the vertices
            for k in range(3):
                IA[k*M:(k+1)*M] = ITris[:, k]
                VA[k*M:(k+1)*M] = 0.5*NMag
    
    #Step 2: Create laplacian matrix
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    #Create the diagonal by summing the rows and subtracting off the nondiagonal entries
    L = sparse.dia_matrix((L.sum(1).flatten(), 0), L.shape) - L
    
    #Step 3: Add anchors
    L = L.tocoo()
    I = L.row.tolist()
    J = L.col.tolist()
    V = L.data.tolist()
    I = I + range(N, N+len(anchorsIdx))
    J = J + anchorsIdx.tolist()
    V = V + [anchorWeights]*len(anchorsIdx)
    L = sparse.coo_matrix((V, (I, J)), shape=(N+len(anchorsIdx), N)).tocsr()
    return L

def solveLaplacianMesh(mesh, anchors, anchorsIdx):
    N = mesh.VPos.shape[0]
    L = getLaplacianMatrixCotangent(mesh, anchorsIdx)
    delta = L.dot(mesh.VPos)
    delta[N:, :] = anchors
    for k in range(3):
        mesh.VPos[:, k] = lsqr(L, delta[:, k])[0]
    mesh.saveFile("out.off")

class DeformationTransferer:
    def __init__(self, origVideo, warpedVideo):
        self.origVideo = origVideo
        self.warpedVideo = warpedVideo
        self.origFrames = self.origVideo.Frames
        self.warpedFrames = self.warpedVideo.Frames
        self.NFrames = self.origFrames.shape[0]
        self.ITris = self.origVideo.ITris
        self.NFaces = self.ITris.shape[0]

        self.count = 0
        self.NVertices = self.origFrames.shape[1]
        self.NVertices4 = self.NVertices + self.NFaces #original vertices plus 1 new vertex (4th vector) for each face
        # Tris4 is Tris plus 4th col indexing 4th vector (which should be mapped to the N to N+F-1 index of VPos4)
        self.Tris4 = np.hstack((self.ITris,
                                    np.reshape(np.arange(self.NVertices, self.NVertices4), (self.NFaces, 1))))
        print "#####debug info: initial values#########"
        print "origFrame shape (NFrames x NVertices x 3):", self.origFrames.shape
        print "warpedFrame shape (NFrames x NVertices x 3): ", self.warpedFrames.shape
        print "ITris shape:", self.ITris.shape
        print "#####end: initial values#########"

    def beginDeformationTransfer(self):
        resultFrames = np.empty([self.NFrames, self.NVertices, 3])  # this is result array to fill in
        resultFrames[0, :, :] = self.warpedFrames[0, :, :]
        origOldVPos4 = self.getVPos4(self.origFrames[0, :, :], self.ITris)  # old VPos with extra NFaces vectors
        warpedOldVPos4 = self.getVPos4(self.warpedFrames[0, :, :], self.ITris)
        for i in range(1, self.NFrames):
            # 1 orig: get newVPos4
            origNewVPos4 = self.getVPos4(self.origFrames[i, :, :], self.ITris)
            # 2 orig: use old and new VPos4 to get S-matrix which shape is 3 x 3NFaces
            S = self.getSMatrix(origOldVPos4, origNewVPos4, self.Tris4)
            # 3 warped: use old VPos4 to get A (coefficient) sparse matrix which shape is 3NFaces x NVertices
            A = self.getAMatrix(warpedOldVPos4, self.Tris4)
            origOldVPos4 = origNewVPos4
            warpedOldVPos4[:, 0] = lsqr(A, S[0, :])[0]
            warpedOldVPos4[:, 1] = lsqr(A, S[1, :])[0]
            warpedOldVPos4[:, 2] = lsqr(A, S[2, :])[0]
           # print "new VPos4 shape:", warpedOldVPos4[np.arange(self.NVertices), :].shape
            resultFrames[i, :, :] = warpedOldVPos4[np.arange(self.NVertices), :]
        self.warpedVideo.Frames = resultFrames



    #get VPos4 (each face has 4 vertices) from VPos3 (each face 3 vertices) with mesh topology given
    def getVPos4(self, VPos3, ITris3):
        V4 = self.get4thVertex(VPos3, ITris3)
        VPos4 = np.vstack((VPos3, V4))
        return VPos4

    # get4thVertex for each face, aka face normal scaled by reciprocal of sqrt of its length
    # (3 vertices's index are stored in every row in ITris)
    def get4thVertex(self, VPos3, ITris3):
        V1 = VPos3[ITris3[:, 1], :] - VPos3[ITris3[:, 0], :]
        V2 = VPos3[ITris3[:, 2], :] - VPos3[ITris3[:, 0], :]
        FNormals = np.cross(V1, V2)

        FNormalsSqrtLength = np.sqrt(np.sum(FNormals**2, 1))[:, None]
        F = FNormals/FNormalsSqrtLength
        Vertex4 = VPos3[ITris3[:, 0], :] + F
        return Vertex4

    def getSMatrix(self, oldVPos4, newVPos4, Tris4):
        v2subv1 = oldVPos4[Tris4[:, 1], :] - oldVPos4[Tris4[:, 0], :]
        v3subv1 = oldVPos4[Tris4[:, 2], :] - oldVPos4[Tris4[:, 0], :]
        v4subv1 = oldVPos4[Tris4[:, 3], :] - oldVPos4[Tris4[:, 0], :]
        tildev2subv1 = newVPos4[Tris4[:, 1], :] - newVPos4[Tris4[:, 0], :]
        tildev3subv1 = newVPos4[Tris4[:, 2], :] - newVPos4[Tris4[:, 0], :]
        tildev4subv1 = newVPos4[Tris4[:, 3], :] - newVPos4[Tris4[:, 0], :]
        assert self.NFaces == Tris4.shape[0]
        S = np.zeros((3, 0))
        for i in range(0, self.NFaces):
            vInv = np.linalg.inv((np.vstack((v2subv1[i, :], v3subv1[i, :], v4subv1[i, :]))).T)
            tildev = (np.vstack((tildev2subv1[i, :], tildev3subv1[i, :], tildev4subv1[i, :]))).T
            S = np.hstack((S, np.dot(tildev, vInv)))
        return S

    def getAMatrix(self, VPos4, Tris4):
        # I, J, and V are parallel numpy arrays that hold the rows, columns, and values of nonzero elements
        I = []
        J = []
        V = []
        v2subv1 = VPos4[Tris4[:, 1], :] - VPos4[Tris4[:, 0], :]
        v3subv1 = VPos4[Tris4[:, 2], :] - VPos4[Tris4[:, 0], :]
        v4subv1 = VPos4[Tris4[:, 3], :] - VPos4[Tris4[:, 0], :]
        assert self.NFaces == Tris4.shape[0]

        for i in range(0, self.NFaces):
            idxRow = i * 3
            vInv = np.linalg.inv((np.vstack((v2subv1[i, :], v3subv1[i, :], v4subv1[i, :]))).T)  # 3x3
            sumOfNegativevInv = np.sum(-1 * vInv, axis = 0) # shape is (3,)
            ###################   ######
            # -A-D-G, A, D, G #   # x1 #
            # -B-E-H, B, E, H # X # x2 #
            # -C-F-I, C, F, I #   # x3 #
            ###################   # x4 #
                                  ######

            # sumOfNegativevInv current looks like this, take care when fill in I, J, V
            ##########################
            # -A-D-G, -B-E-H, -C-F-I #
            ##########################
            for j in range(0, 3):
                I.append(idxRow + j)
                J.append(Tris4[i, 0])
                V.append(sumOfNegativevInv[j])
            # vInv current looks like this. Same, be careful.
            ###########
            # A, B, C #
            # D, E, F #
            # G, H, I #
            ###########
            for j in range(0, 3):
                for k in range(0, 3):
                    I.append(idxRow + k)
                    J.append(Tris4[i, j + 1])
                    V.append(vInv[j, k])
        A = sparse.coo_matrix((V, (I, J)), shape = (3 * self.NFaces, self.NVertices4)).tocsr()
        return A


if __name__ == '__main__':
    #Load in BU bnd files
    buVideo = VideoMesh()
    buVideo.initBUVideo(["bu3/F0012/F0012_AN01WH_F3D.bnd", "bu3/F0012/F0012_HA04WH_F3D.bnd"])
    NFrames = buVideo.Frames.shape[0]
    
    #Load in basel mesh
    baselVertsFile = "BaselVerts.mat"
    ITris = sio.loadmat("BaselTris.mat")['ITris']
    VPos = sio.loadmat(baselVertsFile)['shape']
    VPos = np.reshape(VPos, [3, len(VPos)/3]).T
    
    #Create basel video placeholder
    baselVideo = VideoMesh()
    baselVideo.initBaselVideo(baselVertsFile, NFrames)
    
    #Do coarse deformation transfer
    T = DeformationTransferer(buVideo, baselVideo)
    T.beginDeformationTransfer()
    
    baselVideo.saveFramesOff("Basel")
