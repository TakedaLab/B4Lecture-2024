import sys
import wave
import numpy as np
import matplotlib.pyplot as plt

def get_filename():
    # get command line arguments
    args=sys.argv   # args=[str(program_file_name.py),str(filename.wav)]

    if len(args)==2:
        return args[1]
    else:
        # if command line arguments are excessive or insufficient, print usage.
        print("usage: main_original.py filename")
        return
    
def get_wavedata(filename):    
    # get Wave_read object
    wave_read_obj=wave.open(filename, mode='rb')

    # get binaly data
    wavedata_bin=wave_read_obj.readframes(-1)

    # convert Wave_read object to ndarray
    if wave_read_obj.getsampwidth()==2:
        wave_ndarray=np.frombuffer(wavedata_bin,dtype='int16')
    elif wave_read_obj.getsampwidth()==4:
        wave_ndarray=np.frombuffer(wavedata_bin,dtype='int32')
    else:
        print("Wave read object sample width error")
        return
    
    return wave_ndarray
    




    

    
def main():
    # get filename
    filename=get_filename()

    # if failure, stop program
    if filename==None:
        return
    
    wavedata=get_wavedata()

    if wavedata==None:
        return


if __name__=='__main__':
    main()
