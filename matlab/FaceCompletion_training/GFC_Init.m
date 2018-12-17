function parm = GFC_Init(Solver)

parm.patchsize = 128;
parm.batchsize = 2;
parm.masksize = 64;  
parm.masksize_x = 100; %(hair bangs)
parm.masksize_y = 64;  %(hair bangs)
parm.interval = 1;
parm.occludertype = 3;   %1 = HAIR; 2 = HAT; 3 = SUNGLASSES ; 4 = FACE MASK

% if parm.occludertype ==3
%     parm.masksize = 32;  
% end

% training
parm.train_folder = Solver.folder_img;

tdir = dir(fullfile(parm.train_folder, '*.jpg'));

if isempty(tdir)
tdir = dir(fullfile(parm.train_folder, '*.png'));
end

parm.train_num = length(tdir);
fprintf('training number: %d.\n',parm.train_num);
parm.trainlst = cell(1,parm.train_num);
for m = 1:parm.train_num
   parm.trainlst{1,m}  = tdir(m).name;
end

end
