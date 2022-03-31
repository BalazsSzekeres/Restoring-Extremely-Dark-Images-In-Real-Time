
addpath('external\matlabPyrTools','external\randomforest-matlab\RF_Reg_C', 'image');

imagefiles = dir('image\*.jpg');      
nfiles = length(imagefiles);    % Number of files found
scoreMAtotal = 0;
scoreNIQEtotal = 0;
nfiles

for i=1:nfiles
   currentfilename = imagefiles(i).name;
   currentimage = imread(currentfilename);
   scoreMA{i} = quality_predict(currentimage);
   scoreNIQE{i} = niqe(currentimage);
   scoreMAtotal = scoreMAtotal + scoreMA{i};
   scoreNIQEtotal = scoreNIQEtotal + scoreNIQE{i};
   images{i} = currentimage;
   i
end

scoreMAavg = scoreMAtotal/nfiles;
scoreNIQEavg = scoreNIQEtotal/nfiles;