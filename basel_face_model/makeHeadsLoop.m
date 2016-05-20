%Generate a random head, render it and export it to PLY file
close all
[model, msz] = load_model();
load('04_attributes');
%%
% Generate a random texture
beta  = randn(msz.n_tex_dim, 1);
tex    = coef2object( beta,  model.texMU,   model.texPC,   model.texEV );
% Rendering parameters
rp     = defrp;
rp.phi = 0.5;
rp.dir_light.dir = [0;1;1];
rp.dir_light.intens = 0.6*ones(3,1);

t = linspace(0, 2*pi, 100);
u1 = randn(msz.n_shape_dim, 1); u1 = sqrt(200)*u1/norm(u1);
u2 = randn(msz.n_shape_dim, 1); u2 = sqrt(200)*u2/norm(u2);
for ii = 1:length(t)
    %alpha = u1*cos(t(ii)) + u2*sin(t(ii)); %Random loop
    %alpha = 4*[cos(t(ii)); sin(t(ii))]; %First 2 PCs loop
    alpha = 50*age_shape(1:199)*cos(t(ii)) + 5*gender_shape(1:199)*sin(t(ii)); %Age Loop
    shape  = coef2object( alpha, model.shapeMU, model.shapePC, model.shapeEV ); %Age + Gender loop
    clf;
    display_face(shape, tex, model.tl, rp);
    print('-dpng', '-r100', sprintf('%i.png', ii));
end

