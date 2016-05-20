buface = load('F0012_Neutral.txt'); YScale = 1e3;
[model, msz] = load_model();
%%
AlphaFrac = 0.5; %Fraction of uniqe corresponding points needed to update model
alpha = randn(msz.n_tex_dim, 1);
beta = randn(msz.n_tex_dim, 1);
tex = coef2object( beta,  model.texMU,   model.texPC,   model.texEV);
shape = coef2object( alpha, model.shapeMU, model.shapePC, model.shapeEV);


shapeSize = size(shape);
size1 = shapeSize(1);
newShape = reshape(shape, 3, size1/3).' / YScale;


%Step 1: Do procrustes on 30 keypoints they share in common to get a good
%initial alignment
m = load('../bu3/basel_and_bu3_mappings.mat');
m = m.mappings;
I = load('../bu3/F0012/F0012_AN01WH_F3D.bnd');
I = I(:, 1)+1;
X = buface(I(m.bu3points), :);
Y = newShape(m.baselpoints, :);
Cx = mean(X, 1);
Cy = mean(Y, 1);
X = bsxfun(@minus, X, Cx);
Y = bsxfun(@minus, Y, Cy);
[U, ~, V] = svd(Y'*X);
R = U*V';
X = (R*X')';
% plot3(X(:, 1), X(:, 2), X(:, 3), '.');
% hold on;
% plot3(Y(:, 1), Y(:, 2), Y(:, 3), 'r.');
buface = bsxfun(@minus, buface, Cx);
buface = (R*buface')';
buface = bsxfun(@plus, buface, Cy);


%Step 2: Perform joint face alignment and model alignment
rp = defrp;
rp.phi = 0.5;
rp.dir_light.dir = [0;1;1];
rp.dir_light.intens = 0.6*ones(3,1);

[faceShape, curCx, curCy, curRx] = icpProjections(buface, newShape, 10, model.shapeMU, model.shapePC, YScale, AlphaFrac, tex, model.tl, rp);
display_face(faceShape/YScale, tex, model.tl, rp);
hold on;
buface = bsxfun(@minus, buface, curCx');
buface = (curRx*buface')';
buface = bsxfun(@plus, buface, curCy');
scatter3(buface(:, 1), buface(:, 3), buface(:, 2), 20, 'fill');

%Step 3: Apply the rigid rotation to the buface keypoints, and find the 
%nearest neighbors in the best fit basel model
X = load('../bu3/F0012/F0012_AN01WH_F3D.bnd');
X = X(:, 2:end);
%First apply initial alignment
X = bsxfun(@minus, X, Cx);
X = (R*X')';
X = bsxfun(@plus, X, Cy);
%Now apply refined alignment
X = bsxfun(@minus, X, curCx');
X = (curRx*X')';
X = bsxfun(@plus, X, curCy');
Y = reshape(faceShape/YScale, [3, length(faceShape)/3])';
idx = knnsearch(Y, X);
figure;
display_face(faceShape/YScale, tex, model.tl, rp);
hold on;
scatter3(Y(idx, 1), Y(idx, 3), Y(idx, 2), 'r', 'fill');
