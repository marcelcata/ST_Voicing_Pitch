function [ X, f0 ] = find_params( frame, rate )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

%%ZERO CROSSING
numzeros=0;
for i=1:length(frame)-1
   if ((frame(i)>0 && frame(i+1)<0) || (frame(i)<0 && frame(i+1)>0))
      numzeros = numzeros + 1;
   end
end
%%CORRELATION
%Normalize and substract mean value
m = mean(frame);
frame = frame - m;
max_value = max(abs(frame));
frame = frame/max_value;
correlation = xcorr(frame);
correlation = correlation((round(length(correlation)/2)):end);
%plot(correlation)
%length(correlation)
%pause
r1_0 = correlation(2)/correlation(1);
%dcorr = diff(correlation);
rmax = find(correlation == max(correlation)); 
%rmin1 = rmin(1);
pitch_max= 200;
pitch_min= 25;
corr_pitch = correlation(pitch_min:pitch_max);
p = find(corr_pitch == max(corr_pitch))+pitch_min;
rP_0 = correlation(p)/correlation(1);
%pause
    
%ENERGY
E=sum(frame'*frame'')/length(frame);

%CREATE FEATURE VECTOR
X = [numzeros, E, rP_0, r1_0];
f0 = p/rate;

end

