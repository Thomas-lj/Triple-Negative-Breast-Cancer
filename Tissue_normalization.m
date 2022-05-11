tumor_path = "Z:\Speciale\Tumor_10x_thumb\";
export_path = "Z:\Speciale\Tumor_10x_thumb_macenko_norm\";
patches = dir(tumor_path);

for i = (50:length(patches))
    mkdir(export_path + patches(i).name)
    imgs = dir(fullfile(tumor_path + patches(i).name));
    for k = (3:length(imgs))
        im_path = imgs(k).folder + "\" + imgs(k).name;
        im = imread(im_path);
        [Inorm, H1, E1] = normalizeStaining(im);
        imwrite(Inorm, fullfile(export_path, patches(i).name, "\norm_" + imgs(k).name))
    end
end
%I1 = imread("Z:\Speciale\Tumor\D E19_Thomas TNBC - included slides TCGA-A2-A0T0-01Z-00-DX1 51F904DA-A4B5-4451-8AEF-58E7EF7651DB svs\OriginalImage_62.tif");
%I2 = imread("Z:\Speciale\Tumor\D E19_Thomas TNBC - included slides TCGA-A2-A0CM-01Z-00-DX1 AC4901DE-4B6D-4185-BB9F-156033839828 svs\OriginalImage_195.tif");

% [Inorm1 H1 E1] = normalizeStaining(I1);
% [Inorm2 H2 E2] = normalizeStaining(I2);

% figure('Name', 'Original image 1'), imshow(I1, []);
% figure('Name', 'Normalized image 1'), imshow(Inorm1, []);
% 
% figure('Name', 'Original image 2'), imshow(I2, []);
% figure('Name', 'Normalized image 2'), imshow(Inorm2, []);
%%
tumor_path = "Z:\Speciale\Tumor_10x_thumb\";
export_path = "Z:\Speciale\Tumor_10x_tumb_rein_norm\";
patches = dir(tumor_path);
target_img = imread("C:\Users\dumle\Desktop\Reinhard_norm\target.tif");

for i = (3:length(patches))
    mkdir(export_path + patches(i).name)
    imgs = dir(fullfile(tumor_path + patches(i).name));
    for k = (3:length(imgs))
        im_path = imgs(k).folder + "\" + imgs(k).name;
        im = imread(im_path);
        Inorm = stainnorm_reinhard(im, target_img);
        imwrite(Inorm, fullfile(export_path, patches(i).name, "\norm_" + imgs(k).name))
    end
end

