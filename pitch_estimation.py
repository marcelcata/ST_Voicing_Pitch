# -*- coding: utf8 -*-

"""
Simple pitch estimation
"""

from __future__ import print_function
import os
import numpy as np
import math
from scipy.io import wavfile, loadmat
from scipy.signal import correlate
#from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#from python_speech_features import mfcc


#BASELINE
def autocorr_method(frame, rate):
    """Estimate pitch using autocorrelation
    """
    defvalue = (0.0, 1.0)

    # Calculate autocorrelation using scipy correlate
    #ho passem a float
    frame = frame.astype(np.float)
    #restem la mitjana
    frame -= frame.mean()
    #busquem el valor màxim del frame
    amax = np.abs(frame).max()
    #normalitzem
    if amax > 0:
        frame /= amax
    else:
        return defvalue

    #la autocorrelacio
    corr = correlate(frame, frame)
    # keep the positive part
    corr = corr[len(corr)/2:]

    # Find the first minimum
    dcorr = np.diff(corr)
    rmin = np.where(dcorr > 0)[0]
    if len(rmin) > 0:
        rmin1 = rmin[0]
    else:
        return defvalue

    # Find the next peak
    peak = np.argmax(corr[rmin1:]) + rmin1
    rmax = corr[peak]/corr[0]
    f0 = rate / peak


    #aquestes son les condicions per decidir si és voiced o unvoiced
    if rmax > 0.6 and f0 > 50 and f0 < 550:
        return f0
    else:
        return 0;

#1st METHOD
def center_clipping(frame, clipping_factor = 0.3):
    frame_max = np.abs(frame).max()
    clipping_value = frame_max * clipping_factor
    for i in range(len(frame)):
        if abs(frame[i]) < clipping_value:
            frame[i] = 0.0
        else:
            if frame[i]>0:
                frame[i] -= clipping_value
            else:
                frame[i] += clipping_value
    return frame
def autocorr_method2(frame, rate):
    """Estimate pitch using autocorrelation
    """
    defvalue = (0.0, 1.0)

    # Calculate autocorrelation using scipy correlate
    frame = frame.astype(np.float)
    frame -= frame.mean()
    amax = np.abs(frame).max()
    if max > 0:
        frame /= amax
    else:
        return defvalue

    #print(len(frame))
    frame2 = center_clipping(frame, clipping_factor=0.4)
    #print(len(frame))
    corr = correlate(frame2, frame2)
    # keep the positive part
    corr = corr[len(corr)/2:]

    # Find the first minimum
    dcorr = np.diff(corr)
    rmin = np.where(dcorr > 0)[0]
    if len(rmin) > 0:
        rmin1 = rmin[0]
    else:
        return 0

    # Find the next peak
    peak = np.argmax(corr[rmin1:]) + rmin1
    rmax = corr[peak]/corr[0]
    f0 = rate / peak

    #conditions to decide if voiced or unvoiced
    if rmax > 0.6 and f0 > 50 and f0 < 550:
        return f0
    else:
        return 0;

#2nd METHOD

#def MFFC_creator(frame,rate):

    #mfcc_feat = mfcc(frame, rate)
    #fbank_feat = logfbank(frame, rate)
    #return mfcc_feat

def zero_crossing(frame, rate):
    n_zero_crossings = sum(
        1 for i in range(0, len(frame) - 1) if frame[i] * next((j for j in list(frame)[i + 1:] if j != 0), 0) < 0)
    #print(n_zero_crossings)
    return n_zero_crossings
def obtain_correlation(frame,rate):
    defvalue = (0.0, 1.0)
    # Calculate autocorrelation using scipy correlate
    frame = frame.astype(np.float)
    frame -= frame.mean()
    amax = np.abs(frame).max()
    if max > 0:
        frame /= amax
    else:
        return defvalue
    corr = correlate(frame, frame)
    # keep the positive part
    corr = corr[len(corr) / 2:]
    return corr
def obtain_features(frame, rate):
    #obtain features
    # MFCCs = MFFC_creator(frame, rate)
    num_zeros = zero_crossing(frame, rate)
    correlation = obtain_correlation(frame, rate)
    dcorr = np.diff(correlation)
    rmin = np.where(dcorr > 0)[0]
    rmin1 = rmin[0]
    p = np.argmax(correlation[rmin1:]) + rmin1
    rp_0 = correlation[p] / correlation[0]
    r1_0 = correlation[1] / correlation[0]
    E=sum(frame**2)/len(frame)
    features = [num_zeros, E, rp_0, r1_0]
    return features
def classifier(options, gui):
    print("Loading data...")
    #mat = loadmat(os.path.join(options.datadir, "db_pda.mat"))  # y
    print("prova 1 successful")
    (dataset, classes) = loadmat(os.path.join(options.datadir, 'db_pda.mat')) #y

    print("prova 2")    #dataset = [] #x
    #print("Start training: ... ...")
    #with open(gui) as f:
    #    for line in f:
    #        line = line.strip()
    #        if len(line) == 0:
    #            continue
    #        filename_f0ref = os.path.join(options.datadir, line + ".f0ref")
    #        filename = os.path.join(options.datadir, line + ".wav")
    #        print("Processing:", filename)
    #        file = open(filename_f0ref)
    #         aux = [float(line.strip()) for line in file]
    #         classes += aux
    #         #print("classes")
    #         #print(classes)
    #         #print(len(classes))
    #         rate, data = wavfile.read(filename)
    #         nsamples = len(data)
    #         # From miliseconds to samples
    #         ns_windowlength = int(round((options.windowlength * rate) / 1000))
    #         ns_framelength = int(round((options.framelength * rate) / 1000))
    #         for ini in range(0, nsamples - ns_windowlength + 1, ns_framelength):
    #             frame = data[ini:ini + ns_windowlength]
    #             dataset.append(obtain_features(frame, rate))
    #         if len(dataset)<len(classes):
    #             classes = classes[:len(dataset)]
    #         else:
    #             dataset = dataset[:len(classes)]
    #         #print("dataset")
    #         #print(dataset)
    #         #print(len(dataset))
    # #make class labels be 0 or 1
    # for i in range(len(classes)):
    #     if classes[i] > 0:
    #         classes[i] = 1
    # file1 = os.path.join(options.datadir, "dataset.txt")
    # file2 = os.path.join(options.datadir, "labels.txt")
    # with open(file1, 'wt') as database:
    #     for line1 in dataset:
    #         print(line1, file = database)
    # with open(file2, 'wt') as labels:
    #     for line2 in classes:
    #         print(line2, file = labels)
    # #X_train, X_test, y_train, y_test = train_test_split(dataset, classes, test_size=.2, random_state=42)
    print("Start training")
    clf = SVC(kernel="linear", C=0.025)
    #clf.fit(X_train, y_train)
    clf.fit(dataset, classes)
    print("Training finished!")
    wav2f0(options,gui, clf = clf)
def method3(frame,rate, clf):
    features = obtain_features(frame, rate)
    prediction = clf.predict(features)
    if prediction == 0:
        return 0
    else:
        return autocorr_method(frame,rate)




def wav2f0(options, gui, method = 1, clf = None):
    with open(gui) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            filename = os.path.join(options.datadir, line + ".wav")
            f0_filename = os.path.join(options.datadir, line + ".f0")
            print("Processing:", filename, '->', f0_filename)
            rate, data = wavfile.read(filename)
            with open(f0_filename, 'wt') as f0file:
                nsamples = len(data)

                # From miliseconds to samples
                ns_windowlength = int(round((options.windowlength * rate) / 1000))
                ns_framelength = int(round((options.framelength * rate) / 1000))
                for ini in range(0, nsamples - ns_windowlength + 1, ns_framelength):
                    frame = data[ini:ini+ns_windowlength]
                    #if method < 1:
                    f0 = autocorr_method(frame, rate)
                    #else:
                    #    clf = 6
                    #    f0 = method3(frame,rate, clf)
                    print(f0, file=f0file)

def main(options, args):
    wav2f0(options, args[0])

if __name__ == "__main__":
    import optparse
    optparser = optparse.OptionParser(
        usage='%prog [OPTION]... FILELIST\n' + __doc__)
    optparser.add_option(
        '-w', '--windowlength', type='float', default=32,
        help='windows length (ms)')
    optparser.add_option(
        '-f', '--framelength', type='float', default=15,
        help='frame shift (ms)')
    optparser.add_option(
        '-d', '--datadir', type='string', default='data',
        help='data folder')

    options, args = optparser.parse_args()

    main(options, args)
