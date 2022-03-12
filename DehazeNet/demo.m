clc;
clear;
close all;

haze=imread('./image/029.jpg');
haze=double(haze)./255;
dehaze=run_cnn(haze);
imshow(dehaze);
imwrite(dehaze, fullfile('./image/','029_no_enhance_cnn.png'))



haze2=imread('./image/029_enhanced.jpg');
haze2=double(haze2)./255;
dehaze2=run_cnn(haze2);
imshow(dehaze2);
imwrite(dehaze2, fullfile('./image/','029_enhanced_cnn.png'))
