close all;
clear all;
dataset = load('db_PDA.mat');
X = dataset.X;
y = dataset.y;
%to label classes as voiced or unvoiced
%voiced = 1, unvoiced = 0
y(y>0) = 1;
windowlength = 32;
framelength = 15;
%naive bayes classifier
clf = fitcnb(X,y)
%clf = fitcsvm(X,y)

for i = 1:50
    if(i<10)
        name = char(string('rl00')+num2str(i)+string('.wav'));
        f0file = char(string('rl00')+num2str(i)+string('.f0'));
    else
        name = char(string('rl0')+num2str(i)+string('.wav'));
        f0file = char(string('rl0')+num2str(i)+string('.f0'));
    end
    [data,rate]=audioread(name);
    nsamples = length(data);
    ns_windowlength = round((windowlength * rate) / 1000);
    ns_framelength = round((framelength * rate) / 1000);
    for ini=1: ns_framelength :(nsamples - ns_windowlength)
        frame = data(ini:ini+ns_windowlength);
        [X,f0]= find_params(frame, rate);
        label= predict(clf,X)
%         if(label == 0) %signal is classified as unvoiced, there's no pitch
%             f0 = 0;
%         end
%        f0 = [f0];
        save (f0file,'f0','-ASCII','-append');        
    end    
end

for i = 1:50
    if(i<10)
        name = char(string('sb00')+num2str(i)+string('.wav'));
        f0file = char(string('sb00')+num2str(i)+string('.f0'));
    else
        name = char(string('sb0')+num2str(i)+string('.wav'));
        f0file = char(string('sb0')+num2str(i)+string('.f0'));
    end
    [data,rate]=audioread(name);
    nsamples = length(data);
    ns_windowlength = round((windowlength * rate) / 1000);
    ns_framelength = round((framelength * rate) / 1000);
    for ini=1: ns_framelength :(nsamples - ns_windowlength)
        frame = data(ini:ini+ns_windowlength);
        [X,f0]= find_params(frame, rate);
%         label = predict(clf,X);
%         if(label == 0) %signal is classified as unvoiced, there's no pitch
%             f0 = 0;
%         end
%         f0 = [f0];
        save (f0file,'f0','-ASCII','-append'); 
    end    
end
