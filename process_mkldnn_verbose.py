import re
import getopt
import sys
import os
import glob 
import pandas as pd

#Inititalize inputs
input_file_name = ""
num_cores = 0
freq = 0 
vector_units_per_core = 0
output = ""

#Read user provided inputs
argv=sys.argv[1:]

try:
    opts,args = getopt.getopt(argv, 'i:c:v:f:o:', ['input_file', 'num_cores', 'vec_units_per_core', 'frequency', 'output'])
    if len(opts) != 5:
        print('usage: python process_mkldnn_verbose.py -i <input file or dir> -c <num_cores> -v <vec_units_per_core> -f <frequency> -o <output summary name>')
        sys.exit(2)
    else:
        for opt, arg in opts:
            if opt == '-i' or opt == '--input':
                input_path = arg
            elif opt == '-c' or opt == '--num_cores':
                num_cores = int(arg)
            elif opt == '-v' or opt == '--vec_units_per_core':
                vector_units_per_core = int(arg)
            elif opt == '-f' or opt == '--frequency':
                freq = float(arg)
            elif opt == '-o' or opt == '--output':
                output = str(arg)
except getopt.GetoptError:
    print('usage: python process_mkldnn_verbose.py -i <input file or dir> -c <num_cores> -v <vec_units_per_core> -f <frequency> -o <output summary name>')
    sys.exit(2)

#change based on platform --> AVX2 - 8 and 32 and AVX512 16 and 64
ideal_macspersec_fp32 = 16*vector_units_per_core*num_cores*freq*1000000000
ideal_macspersec_int8 = 64*vector_units_per_core*num_cores*freq*1000000000

print("CORE COUNT: {c} VECTOR UNITS PER CORE: {v} FREQUENCY: {f}".format(c=num_cores, v=vector_units_per_core, f=freq))

#store a nested dictionary for each of the kernels
kernel_bag = {} 

#check if input_path is file or dir
if os.path.isdir(input_path):
    input_path = input_path+"/*.txt"

for input_file in glob.glob(input_path):
    file_kernel = {} 

    print("Reading file %s" %input_file)
    output_file_name = input_file.split('.csv')[0] + '_processed.csv'
    summary_file_name = input_file.split('.csv')[0] + '_summary.csv'
    print("--Writing file %s" %output_file_name)
    print("--Writing file %s" %summary_file_name)

    input_file = open(input_file, mode='r')
    output_file = open(output_file_name, mode='w')

    #write the header rows for the output file
    header = ['DNNL','Run','Device','Operator','ISA','','','','','','actual_time_ms','mb','g','ic','oc','ih','oh','kh','sh','dh','ph','iw','ow','kw','sw','dw','pw','is_int8','MACs','ideal_time_ms','efficiency']
    header_str = ""
    for h in header:
        header_str += str(h) + ","
    header_str += "\n"
    output_file.write(header_str)

    for line in input_file:
        columns = line.split('\n')[0].split('\r')[0].split(',')
        # changing from dnnl_verbose to exec for the check since first 3 rows are info garbage
        if not 'dnnl_verbose' in columns[0]:
            continue
        if 'info' in columns[1]:
            continue
 
        regex1 = re.compile(r'mb(?P<mb>[0-9]+)*_ic(?P<ic>[0-9]+)*oc(?P<oc>[0-9]+)*_ih(?P<ih>[0-9]+)*oh(?P<oh>[0-9]+)*kh(?P<kh>[0-9]+)*sh(?P<sh>[0-9]+)*dh(?P<dh>[0-9]+)*ph(?P<ph>[0-9]+)*_iw(?P<iw>[0-9]+)*ow(?P<ow>[0-9]+)*kw(?P<kw>[0-9]+)*sw(?P<sw>[0-9]+)*dw(?P<dw>[0-9]+)*pw(?P<pw>[0-9]+)*')
        regex2 = re.compile(r'mb(?P<mb>[0-9]+)*_g(?P<g>[0-9]+)ic(?P<ic>[0-9]+)*oc(?P<oc>[0-9]+)*_ih(?P<ih>[0-9]+)*oh(?P<oh>[0-9]+)*kh(?P<kh>[0-9]+)*sh(?P<sh>[0-9]+)*dh(?P<dh>[0-9]+)*ph(?P<ph>[0-9]+)*_iw(?P<iw>[0-9]+)*ow(?P<ow>[0-9]+)*kw(?P<kw>[0-9]+)*sw(?P<sw>[0-9]+)*dw(?P<dw>[0-9]+)*pw(?P<pw>[0-9]+)*')
        regex3 = re.compile(r'g(?P<g>[0-9]+)mb(?P<mb>[0-9]+)*_ic(?P<ic>[0-9]+)*oc(?P<oc>[0-9]+)*_ih(?P<ih>[0-9]+)*oh(?P<oh>[0-9]+)*kh(?P<kh>[0-9]+)*sh(?P<sh>[0-9]+)*dh(?P<dh>[0-9]+)*ph(?P<ph>[0-9]+)*_iw(?P<iw>[0-9]+)*ow(?P<ow>[0-9]+)*kw(?P<kw>[0-9]+)*sw(?P<sw>[0-9]+)*dw(?P<dw>[0-9]+)*pw(?P<pw>[0-9]+)*')
        final_columns = columns
        
        # process only convolutions
        if columns[3] == 'convolution':
            #extract
            ptrn = columns[9]       #kernel column 
            is_int8 = 'int8' in columns[4]  #jit_int8:* is in the 5th column
            match = regex1.match(ptrn)
            if match:
                conv_params = match.groupdict()
                conv_params['g'] = 1
            else:
                match = regex2.match(ptrn)
                if match:
                    conv_params = match.groupdict()
                else: 
                    match = regex3.match(ptrn)          #terrible bandaid fix for short term testing
                    conv_params = match.groupdict()
                
            actual_time_ms = float(columns[10])

            #compute
            macs = int(conv_params['mb'])*int(conv_params['ic'])*int(conv_params['oc'])*int(conv_params['oh'])*int(conv_params['ow'])*int(conv_params['kh'])*int(conv_params['kw'])/int(conv_params['g'])
            ideal_time = 0
            if is_int8:
                ideal_time_ms = (macs/ideal_macspersec_int8)*1000
            else:
                ideal_time_ms = (macs/ideal_macspersec_fp32)*1000
            efficiency = ideal_time_ms/actual_time_ms*100
            
            #get isa
            isa = "unknown"
            if "avx512" in columns[4]:
                isa = "AVX512"
            elif "avx2" in columns[4]:
                isa = "AVX2"
            elif "sse" in columns[4]:
                isa = "SSE"
            elif "uni" in columns[4]:
                isa = "UNI"
            elif "gemm" in columns[4]:
                isa = "GEMM"

            #output
            final_columns.append(conv_params['mb'])
            final_columns.append(conv_params['g'])
            final_columns.append(conv_params['ic'])
            final_columns.append(conv_params['oc'])
            final_columns.append(conv_params['ih'])
            final_columns.append(conv_params['oh'])
            final_columns.append(conv_params['kh'])
            final_columns.append(conv_params['sh'])
            final_columns.append(conv_params['dh'])
            final_columns.append(conv_params['ph'])
            final_columns.append(conv_params['iw'])
            final_columns.append(conv_params['ow'])
            final_columns.append(conv_params['kw'])
            final_columns.append(conv_params['sw'])
            final_columns.append(conv_params['dw'])
            final_columns.append(conv_params['pw'])
            final_columns.append(int(is_int8))
            final_columns.append(macs)
            final_columns.append(ideal_time_ms)
            final_columns.append(efficiency)

            #LINE LEVEL check if the kernel already exists in the file 
            unique_key = isa + str(is_int8) + ptrn
            if unique_key in file_kernel:
                #update the file level kernel
                #TODO: Have to account for dtype and if int8
                file_kernel[unique_key]["total_time_ms"] = file_kernel[unique_key]["total_time_ms"] + actual_time_ms
                file_kernel[unique_key]["exec_cnt"] = file_kernel[unique_key]["exec_cnt"] + 1
            else: 
                #create a new one for the line level kernel
                kernel = {}
                kernel["kernel"] = ptrn
                kernel["total_time_ms"] = actual_time_ms
                kernel["ideal_time_ms"] = ideal_time_ms
                kernel["exec_cnt"] = 1
                kernel["isa"] = isa
                kernel["macs"] = macs 
                kernel["int8"] = is_int8
                kernel["model_cnt"] = 1
                #add kernel to the list of kernel in the file
                file_kernel[unique_key] = kernel 
            
        output_str = ''
        for x in final_columns:
            output_str += str(x) + ','
        output_str += '\n'
        output_file.write(output_str)
    
    #if the file did not have any conv (empty file, corrupt, etc.) skip
    if not file_kernel:
        continue

    #print the file kernel 
    file_df = pd.DataFrame.from_dict(file_kernel, orient="index")
    #calculate the average time and efficiency 
    file_df["avg_time_ms"] = file_df["total_time_ms"] / file_df["exec_cnt"]
    file_df["efficiency"] = file_df["ideal_time_ms"] / file_df["avg_time_ms"] * 100.0
    file_df = file_df.sort_values(["avg_time_ms"], ascending=False)
    file_df.to_csv(summary_file_name, sep=",")

    #transfer the kernel information into the kernel_bag for entire runs of multiple files
    #these are the kernels from 1 file (model) and placing them in the bag 
    for k, v in file_kernel.items():
        if k in kernel_bag: 
            kernel_bag[k]["total_time_ms"] = kernel_bag[k]["total_time_ms"] + file_kernel[k]["total_time_ms"]
            kernel_bag[k]["exec_cnt"] = kernel_bag[k]["exec_cnt"] + file_kernel[k]["exec_cnt"]
            kernel_bag[k]["model_cnt"] = kernel_bag[k]["model_cnt"] + file_kernel[k]["model_cnt"] #the latter should always be 1
        else: 
            kernel_bag[k] = file_kernel[k]

    input_file.close()
    output_file.close()

#finally, process all of the kernels for overall summary
df = pd.DataFrame.from_dict(kernel_bag, orient="index")
print(df)
df["avg_time_ms"] = df["total_time_ms"] / df["exec_cnt"]
df["efficiency"] = df["ideal_time_ms"] / df["avg_time_ms"] * 100.0
df = df.sort_values(["avg_time_ms"], ascending=False)
print("CORE COUNT: {c} VECTOR UNITS PER CORE: {v} FREQUENCY: {f}".format(c=num_cores, v=vector_units_per_core, f=freq))

if output:
    output = output + "-" + str(num_cores) + "cores_" + "_" + str(vector_units_per_core) + "vecUnits_" + str(freq) + "freq.csv"
else: 
    output = "MKLDNN_ALL_SUMMARY.csv" + "-" + str(num_cores) + "cores_" + "_" + str(vector_units_per_core) + "vecUnits_" + str(freq) + "freq.csv"

df.to_csv(output, sep=",", index=False)

print("Processing completed!")

