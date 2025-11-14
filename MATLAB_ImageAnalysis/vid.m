clc, clear all, close all

fps = 1;
minutes = 1; %stapgrootte frames (in min)
sc = 60*minutes*fps; %stapgrootte frames (in sec)

filePattern_image = fullfile('*.jpg');
theFiles_image = dir(filePattern_image);
q=1;

for lengtelijst = 1:numel(theFiles_image)
    if theFiles_image(lengtelijst).bytes > 5000
        Files_image{q, 1} = theFiles_image(lengtelijst).name;
        q = q+1;
    end
end
image_all = cell(1, length(Files_image));

filePattern_video = fullfile('*.avi');
theFiles_video = dir(filePattern_video);
q=1;
for lengtelijst = 1:numel(theFiles_video)
    if theFiles_video(lengtelijst).bytes > 5000
        Files_video{q, 1} = theFiles_video(lengtelijst).name;
        q = q+1;
    end
end
video_all = cell(1, length(theFiles_video));
videos = cell(1, numel(theFiles_video));

%video inladen
for q = 1:numel(Files_video)
video{q, 1} = VideoReader(Files_video{q});
frame{q, 1} = imread(Files_image{q}); %frame 1, before water is applied
imrotated{q, 1} = imrotate(frame{q, 1}, 90);
end
totalFrames = video{1}.NumFrames;

for qt = 1:numel(Files_video)
    q = 1;
    for l = 1:sc:totalFrames
        q=q+1;
        frames = read(video{qt, 1}, l);
        frame{qt, q} = frames;
        imrotated{qt, q} = imrotate(frame{qt, q}, 90);
    end
end


%% Variables for image analysis
[rows, columns, rgb] = size(imrotated{1});
t = 10; %step size
backgroundnoise = (graythresh(imrotated{1})*100)*1.1; % 10% error marge
cmnsbegin = 150; %starting column for measurement [1 - cmnsend]
cmnsend = columns-200; %last column for measurement [1 - columns]
rowbegin = 850; %starting row to determine deformation of paper [1 - rowend]
rowend = rows-200; %last row to determine deformation of paper [1 - rows]

lim = 50; %limit of reasonable pixel displacement
n1_error = 0.1; %error marge for width determination
pixelpermm = 25.4498330096883;
mmperpixel = 1/pixelpermm; %mm/pixel of current set up

threshold = mean(mean(imrotated{1}(rowbegin:rowend, ...
    cmnsbegin+(((cmnsend-cmnsbegin)/2)-5):cmnsbegin+(((cmnsend-cmnsbegin)/2)+5)))); %threshold for vertical lines of image

x = cmnsbegin:cmnsend;
y = rowbegin:rowend;

droprate = 0.1; % in ml/min
platform_moved = 100; % in mm
platform_velocity = 2; % in mm/s
volume_water = (platform_moved / platform_velocity) * (droprate/60); % in ml

%% Displacement of laser
imtool close all
for q = 1:numel(Files_image)
    for b = 1:length(imrotated)
        k = 0;
        for j = cmnsbegin:cmnsend
            if t ~= 1
                if  round(j-t/2) < cmnsbegin
                    c = j:j+t;
                elseif round(j+t/2) > cmnsend
                    c = j:cmnsend;
                else
                    c = round(j-t/2):round(j+t/2);
                end
            elseif t == 1
                if  j < cmnsend-t
                    c = j:j+t;
                else
                    c = j:cmnsend;
                end
            end
            k=k+1;
            averageimg = (sum(imrotated{q, b}(:, c), 2))/(numel(c)); %takes averages of [c] columns

            if mean(averageimg) > threshold %removes vertical lines (unwanted)
                averageimg = 0;
            end

            newimg(:, k) = averageimg; %new image without vertical lines
            onepeak = newimg(rowbegin:rowend, k); %shows image without unwanted light of the source
            n = 0.5*max(onepeak); %determine intensity value

            if n < backgroundnoise %remove values for interpolation
                n = 0;
            elseif onepeak == 0
                n = 0;
            else
                n;
            end

            tot_n(k, b) = n;

            %interpolation
            if n > 0
                [max_y{q}(b), max_x{q}(b)] = max(onepeak);
                [left_index{q}(b), leftindex{q}(b)] = min(abs(onepeak(1:max_x{q}(b)) - n));
                [right_index{q}(b), rightindex{q}(b)] = min(abs(onepeak(max_x{q}(b)+1:rowend-rowbegin) - n));
                leftindex_new{q}(b) = leftindex{q}(b);
                rightindex{q}(b) = rightindex{q}(b) + max_x{q}(b);
                rightindex_new{q}(b) = rightindex{q}(b);

                if onepeak(leftindex{q}(b)) < n
                    leftindex_new{q}(b) = leftindex{q}(b) + 1;
                    while onepeak(leftindex_new{q}(b)) == onepeak(leftindex{q}(b))
                        leftindex_new{q}(b) = leftindex_new{q}(b) + 1;
                    end
                elseif onepeak(leftindex{q}(b)) > n
                    leftindex_new{q}(b) = leftindex{q}(b) - 1;
                    while onepeak(leftindex_new{q}(b)) == onepeak(leftindex{q}(b))
                        leftindex_new{q}(b) = leftindex_new{q}(b) - 1;
                    end
                elseif onepeak(leftindex{q}(b)) == n
                    leftindex_new{q}(b) = leftindex{q}(b) - 1;
                    while onepeak(leftindex_new{q}(b)) == onepeak(leftindex{q}(b))
                        leftindex_new{q}(b) = leftindex_new{q}(b) - 1;
                    end
                end

                if onepeak(rightindex{q}(b)) < n
                    rightindex_new{q}(b) = rightindex{q}(b) - 1;
                    while onepeak(rightindex_new{q}(b)) == onepeak(rightindex{q}(b))
                        rightindex_new{q}(b) = rightindex_new{q}(b) - 1;
                    end
                elseif onepeak(rightindex{q}(b)) > n
                    rightindex_new{q}(b) = rightindex{q}(b) + 1;
                    while onepeak(rightindex_new{q}(b)) == onepeak(rightindex{q}(b))
                        rightindex_new{q}(b) = rightindex_new{q}(b) + 1;
                    end
                elseif onepeak(rightindex{q}(b)) == n
                    rightindex_new{q}(b) = rightindex{q}(b) + 1;
                    while onepeak(rightindex_new{q}(b)) == onepeak(rightindex{q}(b))
                        rightindex_new{q}(b) = rightindex_new{q}(b) + 1;
                    end
                end

                if n > backgroundnoise
                    leftside{q}(k, b) = interp1([onepeak(leftindex_new{q}(b)) onepeak(leftindex{q}(b))], [leftindex_new{q}(b) leftindex{q}(b)], n);
                    rightside{q}(k, b) = interp1([onepeak(rightindex_new{q}(b)) onepeak(rightindex{q}(b))], [rightindex_new{q}(b) rightindex{q}(b)], n);
                    center{q}(k, b) = mean([leftside{q}(b) rightside{q}(b)]);
                else
                    leftside{q}(k, b) = 0;
                    rightside{q}(k, b) = 0;
                    center{q}(k, b) = 0;
                end
            else
                leftside{q}(k, b) = NaN;
                rightside{q}(k, b) = NaN;
                center{q}(k, b) = NaN;

            end

            displacement{q}(k, b) = center{q}(k, b) - center{q}(k, 1);
            displacement_mm{q}(k, b) = displacement{q}(k, b) .* mmperpixel;
        end

        displacement{q}(displacement{q}<-lim) = NaN; %limit of reliable values in pixels
        displacement{q}(displacement{q}>lim) = NaN;

%         maximum{q}(b) = max(displacement{q}(:, b));
%         minimum{q}(b) = min(displacement{q}(:, b));
    end
    

    %change value to nan if neighbours are also nan
%     for g = 2:length(displacement)-1
%         if isnan(displacement(g-1, 1)) && isnan(displacement(g+1, 1))
%             displacement(g, 1) = NaN;
%         end
%     end
%     %changes value of other measurement to nan if zero measurement is also nan at x
%     for v = 1:length(displacement)
%         if isnan(displacement(v, 1))
%             displacement(v, :) = NaN;
%         end
%     end

  
  %width determenation
    tot_n(tot_n==0) = NaN;
    avg_n1 = mean(tot_n(:, 1), "omitnan");
    for u = 2:b
        more_gray{q}{:, u} = find(tot_n(:, u) < avg_n1 * (1 - n1_error)  & ~isnan(tot_n(:, 1)) & ~isnan(displacement{q}(:, u)));

        f=1;
        while f < numel(more_gray{q}{:, u}) %remove unreliable data points
            if more_gray{q}{:, u}(f+1, 1) - more_gray{q}{:, u}(f, 1) > t
                more_gray{q}{:, u}(f, 1) = 0;
            end
            f=f+1;
        end
        more_gray{q}{:, u} = more_gray{q}{:, u}(more_gray{q}{:, u} ~= 0);

        if isempty(more_gray{q}{:, u})
            width{q}(:, u-1) = 0;
        else
            width{q}(:, u-1) = max(x(more_gray{q}{:, u})) - min(x(more_gray{q}{:, u}));
        end
    end
end

    %% Plotting
    close all
    f = figure('units','normalized','outerposition',[0 0 1 1]);
    legend_names = cell(1, numel(imrotated));
    legend()

%     for a = 2:numel(legend_names)
    for a = [2, 7, 27]
        plot(x, displacement(:, a));
        hold on
        legend_names{a-1} =  (a-2)*minutes;
        legenda(a-1) = sprintf("t = %d min", legend_names{a-1});
        legappend(legenda(a-1))
        %     pause(1)
        %     exportgraphics(f, "animation_met1.gif", "append", true)
    end
% 
%     hold on
% pos1_min = min(x(more_gray{:, 2}))-cmnsbegin;
% pos1_max = max(x(more_gray{:, 2}))-cmnsbegin;
% plot(pos1_min+cmnsbegin, [displacement(pos1_min, 2)-2.5:displacement(pos1_min, 2)+2.5], "o")
% plot(pos1_max+cmnsbegin, [displacement(pos1_max, 2)-2.5:displacement(pos1_max, 2)+2.5], "o")
% hold on
% pos2_min = min(x(more_gray{:, 7}))-cmnsbegin;
% pos2_max = max(x(more_gray{:, 7}))-cmnsbegin;
% plot(pos2_min+cmnsbegin, [displacement(pos2_min, 7)-2.5:displacement(pos2_min, 7)+2.5], "o")
% plot(pos2_max+cmnsbegin, [displacement(pos2_max, 7)-2.5:displacement(pos2_max, 7)+2.5], "o")


%     
% hold on
% pos1_x = min(x(more_gray{:, 2}))-cmnsbegin;
% plot(pos1_x+cmnsbegin, [displacement(pos1_x, 2)-5:displacement(pos1_x, 2)+5], "o")
% plot(pos1_x, 0, "o")

% legend off
% ylim([-20 60])
% x1 = 300;
% x2 = 410;
% x3 = 550;
% x4 = 690;
% x5 = 800;
% 
% xline(x1, "color", "blue")
% xline(x2, "color", "green")
% xline(x3, "color", "cyan")
% xline(x4, "color", "red")
% xline(x5, "color", "magenta")
% yline(0)
% grid on
% 
% 
% figure(23523)
% for trigger = 2:numel(imrotated)
%     plot(trigger-2, displacement(x1-cmnsbegin, trigger), "o", "color", "blue")
%         hold on
%     plot(trigger-2, displacement(x2-cmnsbegin, trigger), "*", "color", "green")
%     plot(trigger-2, displacement(x3-cmnsbegin, trigger), "*", "color", "cyan")
%     plot(trigger-2, displacement(x4-cmnsbegin, trigger), "x", "color", "red")
%     plot(trigger-2, displacement(x5-cmnsbegin, trigger), "x", "color", "magenta")    
% end
% yline(0)
% grid on
% xlim([40 60])


%% length of the paper
close(figure(12412312))
[size_row, size_column] = size(displacement{1});
% sqrt(x^2+y^2) = schuine zijde lengte
for q = 1:numel(Files_video)
for dt = 2:size_column
    lengte{q}(dt-1) = 0;
    for d = 1:size_row-1

        if ~isnan(displacement{q}((d+1), dt)) && ~isnan(displacement{q}(d, dt)) 
            dx = d+1 - d;
            dy = displacement{q}(d+1, dt) - displacement{q}(d,dt);
            lengte{q}(dt-1) = lengte{q}(dt-1) + sqrt(dx.^2+dy.^2);

        elseif isnan(displacement{q}((d+1), dt)) && ~isnan(displacement{q}(d, dt))
            d_last = d;
            while isnan(displacement{q}(d+1, dt)) && d < size_row-1
                d=d+1;
            end
            if ~isnan(displacement{q}(d+1, dt)) && d < size_row-1
            dx = d+1 - d_last;
            dy = displacement{q}(d+1, dt) - displacement{q}(d_last,dt);
            lengte{q}(dt-1) = lengte{q}(dt-1) + sqrt(dx.^2+dy.^2);
            else
                lengte{q}(dt-1) = lengte{q}(dt-1);
            end
        else
            lengte{q}(dt-1) = lengte{q}(dt-1);

        end  
    end
end
end
timeslot = (0:sc:totalFrames-1)/60;
colors = distinguishable_colors(numel(Files_image));

for q = 1:numel(lengte{q})
figure(12412312), plot(timeslot, lengte{q}, "-o", "color", colors(q, :)), ylabel("length of paper [mm]")
hold on
% hold on, yyaxis right, plot(width{q}, "-x", "color", "red"), ylabel("width of water")
xlabel("time [min]") 
grid on
end

%%
figure(1254)
plot(x, displacement(:, 3), "o")
hold on
plot(x(more_gray{3}), 0, "*")

