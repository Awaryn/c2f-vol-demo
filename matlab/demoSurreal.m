%% Visualization for Human3.6M sample data
% We assume that the network has already been applied on the Human3.6M sample images.
% This code reads the network predictions and visualizes them.
% The demo sequence is Posing_1 from Subject 9 and from camera with code 55011271.

clear; startupSurreal;

% define paths for data and predictions
datapath = '../data/surreal-sample/';
predpath = '../exp/surreal-sample/';
annotfile = sprintf('%sannot/val.h5',datapath);

video_ids = hdf5read(annotfile, 'video_id');
frame_ids = hdf5read(annotfile, 'frame_id');
centers = hdf5read(annotfile, 'center');
scales = hdf5read(annotfile, 'scale');
sgts = hdf5read(annotfile, 'sgt');
parts = hdf5read(annotfile, 'part');

loaded_video = -1;
video_list = importdata(sprintf('%s/val/list.txt',datapath));

% define the reconstruction from the volumetric representation
% if recType = 1, we use the groundtruth depth of the root joint
% if recType = 2, we estimate the root depth based on the subject's skeleton size
% if recType = 3, we estimate the root depth based on the training subjects' mean skeleton size
recType = 2;

% volume parameters
outputRes = 32;     % x,y resolution
depthRes = 32;      % z resolution
numKps = 24;        % number of joints

% main loop to read network output and visualize it
nPlot = 3;
angle = 0;
h = figure('position',[300 300 200*nPlot 200]);
for img_i = 1:length(frame_ids)
    
    % read input info
    frame_id = frame_ids(img_i);
    video_id = video_ids(img_i);
    center = centers(:, img_i);
    scale = scales(img_i);
    Sgt = sgts(:, :, img_i) * 2000.0;
    
    % Load the video if necessary
    if loaded_video ~= video_id
        v = VideoReader(sprintf('%s/val/%s.mp4', datapath, video_list{video_id+1}));
        loaded_video = video_id
    end
    
    I = read(v, frame_id+1);
    K = getIntrinsicBlender()
    
%     imgname = annot.imgname{img_i};
%     center = annot.center(img_i,:);
%     scale = annot.scale(img_i);
%     Sgt = squeeze(annot.S(img_i,:,:));
%     K = annot.K{img_i};
    
    Lgt = limbLength(Sgt,skel);
    zroot = Sgt(3,1);
    bbox = getHGbbox(center,scale);
%     I = imread(sprintf('%s/images/%s.jpg',datapath,imgname));
    img_crop = cropImage(I,bbox);
    
    % read network's output
    joints = hdf5read([predpath 'val_' num2str(img_i)  '.h5'],'preds3D');
    % pixel location
    W = maxLocation(joints(1:2,:),bbox,[outputRes,outputRes]);
    % depth (relative to root)
    Zrel = Zcen(joints(3,:));
    
    % reconstruct 3D skeleton
    if recType == 1
        S = estimate3D(W,Zrel,K,zroot);
    elseif recType == 2
        S = estimate3D(W,Zrel,K,Lgt,skel);
    elseif recType == 3
        S = estimate3D(W,Zrel,K,Ltr,skel);
    end
    
    % visualization
    clf;
    % image
    subplot('position',[0/nPlot 0 1/nPlot 1]);
    imshow(img_crop); hold on;
    % 3D reconstructed pose
    subplot('position',[1/nPlot 0 1/nPlot 1]);
    vis3Dskel(S,skel);
    % 3D reconstructed pose in novel view
    subplot('position',[2/nPlot 0 1/nPlot 1]);
    vis3Dskel(S,skel,'viewpoint',[-angle 0]);
    angle = angle + 10
%     camroll(10);
    pause(0.01);
    
end