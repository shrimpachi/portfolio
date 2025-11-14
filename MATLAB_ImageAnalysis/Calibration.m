clc, clear all, close all
%% call data
T = readtable("Meetdata kalibratie.xlsx", "Sheet","angle 2022mar01");

%table to specific group
% Line5 = [T(:, [2 4 12])];
% Line6 = [T(:, [2 5 13])];
% Line7 = [T(:, [2 6 14])];

%% NPP
figure()
subplot(3, 1, 1)
NPP.line5 = normplot(Line5.theta5_deg_);
title('Line 5 angle')

subplot(3,1,2)
NPP.line6 = normplot(Line6.theta6_deg_);
title('Line 6 angle')

subplot(3,1,3)
NPP.line7 = normplot(Line7.theta7_deg_);
title('Line 7 angle')

%% Variantiecoefficient
%mean values
Mean.Line5 = mean(Line5.theta5_deg_);
Mean.Line6 = mean(Line6.theta6_deg_);
Mean.Line7 = mean(Line7.theta7_deg_);

%standard deviation
s.Line5 = sqrt((sum((Line5.theta5_deg_ - Mean.Line5).^2))/((numel(Line5.theta5_deg_)-1)));
s.Line6 = sqrt((sum((Line6.theta6_deg_ - Mean.Line6).^2))/((numel(Line6.theta6_deg_)-1)));
s.Line7 = sqrt((sum((Line7.theta7_deg_ - Mean.Line7).^2))/((numel(Line7.theta7_deg_)-1)));

%coefficient
VC.Line5 = Mean.Line5/s.Line5;
VC.Line6 = Mean.Line6/s.Line6;
VC.Line7 = Mean.Line7/s.Line7;

%% Residual analysis
Data5 = table2array([T(1:26, 8)]);
Data6 = table2array([T(29:46, 8)]);
Mean_5 = mean(Data5);
Mean_6 = mean(Data6);
res.Line5 = [Data5 - Mean_5];
res.Line6 = [Data6 - Mean_6];
% res.Line5 = [Line5.theta5_deg_ - Mean.Line5];
% res.Line6 = [Line6.theta6_deg_ - Mean.Line6];
% res.Line7 = [Line7.theta7_deg_ - Mean.Line7];

subplot(2,1,1)
plot(res.Line5, 'x');
refline(0, 0)
ylim([-0.05, 0.05])
title('Residual Analysis Line 5')

subplot(2,1,2)
plot(res.Line6, 'x');
refline(0, 0)
ylim([-0.05, 0.05])
title('Residual Analysis Line 6')

% subplot(3,1,3)
% plot(res.Line7, 'x');
% refline(0, 0)
% ylim([-0.4, 0.4])
% title('Residual Analysis Line 7')

%% ANOVA
% ANOVA = anova1([Line5.theta5_deg_, Line6.theta6_deg_, Line7.theta7_deg_]);
close all

liniaal.line5 = Line5.theta5_deg_(1:5, :);
meter.line5 = Line5.theta5_deg_(6:10, :);
tot.line5 = Line5.theta5_deg_(1:10, :);

liniaal.line6 = Line6.theta6_deg_(1:5, :);
meter.line6 = Line6.theta6_deg_(6:10, :);
tot.line6 = Line6.theta6_deg_(1:10, :);

liniaal.line7 = Line7.theta7_deg_(1:5, :);
meter.line7 = Line7.theta7_deg_(6:10, :);
tot.line7 = Line7.theta7_deg_(1:10, :);

ANOVA.Line5 = anova1([liniaal.line5, meter.line5]);
title('ANOVA Line 5')
ANOVA.Line6 = anova1([liniaal.line6, meter.line6]);
title('ANOVA Line 6')
ANOVA.Line7 = anova1([liniaal.line7, meter.line7]);
title('ANOVA Line 7')

% p-value depends on measuring. 
% Lower value probably line thicker, not measured well.
% Another issue might be that the paper does not lie flat (labjack not
% flat, takes holes while drawing lines)

%% Lijn dikte bepalen

