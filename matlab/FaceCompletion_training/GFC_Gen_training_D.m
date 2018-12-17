function [batch_G, batch_D, gt_D, mask] = GFC_Gen_training_D(parm)
% data layer
batch_G = single(zeros(parm.patchsize,parm.patchsize,3,parm.batchsize));
batch_D = single(zeros(parm.patchsize,parm.patchsize,3,parm.batchsize));
gt_D = single(ones(1, 1, 1, parm.batchsize));

idpool = randperm(parm.train_num);
count = 1;
parm.interval = 1;
mask = zeros (parm.interval^2,2,parm.batchsize);

while count <= parm.batchsize
   idx = idpool(count);
   img = imread(fullfile(parm.train_folder,parm.trainlst{idx}));
   img = (imresize(img, [parm.patchsize,parm.patchsize]));

   [r,c,cha] = size(img);
   
   if cha < 3
       img = repmat(img, [1 1 3]);
   end
  
   rate =  (rand - 0.5)/10;
   shift_x = floor(max(r,c) * rate);
   rate =  (rand - 0.5)/10;
   shift_y = floor(max(r,c) * rate);
   scale_x = 1;
   scale_y =  scale_x;
   angle = (rand - 0.5)*(10/180)*pi;
   A = [scale_x * cos(angle), scale_y * sin(angle), shift_x;...
    -scale_x * sin(angle), scale_y * cos(angle), shift_y]';
   T = maketform('affine', A);
   simg = single(imtransform(img, T, 'XData',[1,c], 'YData',[1,r], 'FillValues',127)); 
   img = simg; img = (img - min(img(:))) ./ (max(img(:))-min(img(:))); img = -1 + 2 * img;
   output = img; img1 = fliplr(img); output1 = img1; 
   
   
  
   
%% ----------Sidra commented below code %------------------
%  margin_x = parm.patchsize - parm.masksize -10; 
%  margin_y =  parm.patchsize - parm.masksize -10;
%    p = 1;
%     for i=4:margin_x/parm.interval:margin_x
%        for j=4:margin_y/parm.interval:margin_y
%            rand_x = ceil(rand * margin_x/parm.interval+i); rand_y =
%            ceil(rand * margin_y/parm.interval+j);
%                                  
%            img(max(1,rand_y):max(1,rand_y)+parm.masksize-1, ...
%                max(1,rand_x):max(1,rand_x)+parm.masksize-1,:) =
%                single(-1+2*rand(parm.masksize,parm.masksize,3));
%            
%            img1(max(1,rand_y):max(1,rand_y)+parm.masksize-1, ...
%                max(1,rand_x):max(1,rand_x)+parm.masksize-1,:) =
%                single(-1+2*rand(parm.masksize,parm.masksize,3));
%            
%            mask(p,1,count) = rand_x;mask(p,1,count+1) = rand_x;
%            mask(p,2,count) = rand_y;mask(p,2,count+1) = rand_y; p = p +
%            1;
%        end
%    end
%    
%    batch_G(:,:,1:3,count) = img; batch_G(:,:,1:3,count+1) = img1;
%--------------Code commented by sidra---------------------------%
   
%% %-----------------------------%
   %Sidra Added the lines below
   filename = parm.trainlst{idx};
   if parm.occludertype ==1
        rand_x = [10, 20, 30, 40, 50]; pos_x = randi(length(rand_x)); rand_x = rand_x(pos_x);
        rand_y = [10, 15, 20]; pos_y = randi(length(rand_y));  rand_y = rand_y(pos_y);  %hair bangs starting position
        OcclusionPath = '../FaceCompletion_testing/HairOcclusion/';
   elseif  parm.occludertype ==2
        rand_x = [1, 20, 30, 40, 50]; pos_x = randi(length(rand_x)); rand_x = rand_x(pos_x);
        rand_y = [1, 3, 5, 7, 10]; pos_y = randi(length(rand_y));  rand_y = rand_y(pos_y);  %hat starting position
        OcclusionPath = '../FaceCompletion_testing/HatOcclusion/';
   elseif parm.occludertype == 3
        rand_x = [20,25,30, 35]; pos_x = randi(length(rand_x)); rand_x = rand_x(pos_x);
        rand_y = [45,50,55]; pos_y = randi(length(rand_y));  rand_y = rand_y(pos_y);  %sunglasses starting position
        OcclusionPath = '../FaceCompletion_testing/SunglassesOcclusion/';
   elseif parm.occludertype ==4
       rand_x = [20,25, 30,35]; pos_x = randi(length(rand_x)); rand_x = rand_x(pos_x);
       rand_y = [60,61,62,63, 64]; pos_y = randi(length(rand_y));  rand_y = rand_y(pos_y);  %Mask starting position
       OcclusionPath = '../FaceCompletion_testing/MaskOcclusion/';
   end
  
   maskedImgName = dir([OcclusionPath '*' filename]);   szmm = size(maskedImgName,1);
   if szmm ==0 
       filename
   end
   maskedImg = imread([OcclusionPath maskedImgName(randi([1 szmm], 1,1)).name]); 
   maskedImg = (imresize(maskedImg, [parm.patchsize,parm.patchsize]));
   simg = single(imtransform(maskedImg, T, 'XData',[1,c], 'YData',[1,r], 'FillValues',127));
   img = simg;
   img = (img - min(img(:))) ./ (max(img(:))-min(img(:)));
   img = -1 + 2 * img; img1 = fliplr(img);
   batch_G(:,:,1:3,count) = img;  %occluded input image SIDRA
   batch_G(:,:,1:3,count+1) = img1; %occluded input image flipped SIDRA
   count = 1; p=1 ; %rand_x = 10; rand_y = 18;
   
 
   mask(p,1,count) = rand_x;mask(p,1,count+1) = rand_x;
   mask(p,2,count) = rand_y;mask(p,2,count+1) = rand_y;
   %-------------------------------%
   
   
   batch_D(:,:,:,count) = output;  %groundtruth SIDRA
   batch_D(:,:,:,count+1) = output1; %flipped ground truth SIDRA
   count = count + 2;
end
end
