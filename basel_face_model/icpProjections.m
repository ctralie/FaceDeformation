%Code by Erin Taylor and Nick Pagliuca, edited by Chris Tralie
%ICP 
%%
function[finalFace, curCx, curCy, curRx] = icpB(X, Y, maxIters, mean, pc, YScale, AlphaFrac, tex, tl, rp)
    curCx = [0; 0; 0];
    curCy = [0; 0; 0];
    curRx = eye(3);
    prevIDX = -1*ones(size(X, 1), 1);
    X = X.';
    Y = Y.';
    listD = [];
    for j=1:20
        for i=1:maxIters
             idx = getCorrespondences(X, Y, curCx, curCy, curRx);
             if length(unique(idx)) > AlphaFrac*size(X, 2)
                 if sum(idx == prevIDX) == length(idx)
                     break;
                 end
             end
             prevIDX = idx;
             [newCx, newCy, newRx, alpha, distsSqr, toKeep] = getProcrustesAlignment(X, Y, idx, mean, pc, YScale);
             listD = [listD, distsSqr];
             curCx = newCx;
             curCy = newCy;
             curRx = newRx;
        end
        plot(X, Y, toKeep, curCx, curCy, curRx, j, tex, tl, rp);
        if length(unique(idx)) < AlphaFrac*size(X, 2)
            disp('Not enough corresponding points...delaying update of face');
        end
        disp('Updating face');
        newY = mean + pc * alpha;
        newYSize = size(newY);
        size1 = newYSize(1);
        Y = reshape(newY, 3, size1/3) / YScale;        
        finalFace = newY;
    end
    semilogy(listD);
    print('-dpng', '-r100', sprintf('%i.png', j));
end

function[] = plot(X, Y, toKeep, Cx, Cy, R, num, tex, tl, rp)
    X = bsxfun(@minus, X, Cx);
    Y = bsxfun(@minus, Y, Cy);
    X = R * X;
    X = X.';
    display_face(Y(:), tex, tl, rp);
    %plot3(Y(:, 1), Y(:, 2), Y(:, 3), 'r.');
    hold on
    scatter3(X(toKeep==0, 1), X(toKeep==0, 3), X(toKeep==0, 2), 20, 'b');
    scatter3(X(toKeep==1, 1), X(toKeep==1, 3), X(toKeep==1, 2), 20, 'r');
    print('-dpng', '-r100', sprintf('%i.png', num));
    clf;
end

function[centroid] = getCentroid(PC)
    centroid = mean(PC, 2);
end

function [Cx, Cy, R, alpha, distsSqr, toKeep] = getProcrustesAlignment(X, Y, idx, mean, pc, YScale)
    Cx = getCentroid(X);
    X = bsxfun(@minus, X, Cx);
    YCorr = Y(:, idx);
    Cy = getCentroid(YCorr);
    YCorr = bsxfun(@minus, YCorr, Cy);
    [U, ~, V] = svd(YCorr * X.');
    R = U * V.';
    XTemp = R*X;
    %For redundant indices, keep only the ones closest to the target
    dists = sum((XTemp-YCorr).^2, 1);
    distsSqr = sum(dists);
    [~, aidx] = sort(idx);
    dists = dists(aidx);
    toKeep = zeros(1, length(dists));
    ii = 1;
    while ii <= length(dists)
        si = ii;
        ei = ii;
        d = dists(ii);
        while idx(aidx(ei)) == idx(aidx(si))
            if dists(ei) < d
                d = dists(ei);
                si = ei;
            end
            ei = ei + 1;
            if ei > length(dists)
                break
            end
        end
        toKeep(aidx(si)) = 1;
        ii = ei;
    end
    fprintf(1, '%i, %i\n', length(unique(idx)), sum(toKeep));
    XCorr = inf*ones(size(Y));
    XCorr(:, idx(toKeep == 1)) = bsxfun(@plus, R*X(:, toKeep == 1), Cy);
    XCorr = XCorr(:);
    idx = 1:length(XCorr);
    idx = idx(~isinf(XCorr));
    fprintf(1, 'Subset of size %i used to update alpha\n', sum(toKeep));
    XCorr = YScale*XCorr(idx);
    XCorr = XCorr - mean(idx);
    alpha = pc(idx, :)'*XCorr;
end

function[idx] = getCorrespondences(X, Y, Cx, Cy, Rx)
    X = bsxfun(@minus, X, Cx);
    Y = bsxfun(@minus, Y, Cy);
    X = Rx * X;
    idx = knnsearch(Y.', X.');
end