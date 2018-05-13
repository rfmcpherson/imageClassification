import subprocess

script_dir = "condor_scripts/"
res_dir = "results/"
path = "/u/richard/p3/class/orl/"

def condor(label, data_dir):

    # Make results dir
    subprocess.call(["mkdir", "-p", res_dir+label])

    # Make bash script
    script = ""
    script += "#!/bin/sh\n" 
    script += "export TERM=xterm-256color\n" 
    script += "export HOME=/u/richard\n" 
    script += "export PATH=/u/richard/torch/install/bin:\$PATH\n" 
    script += "export LD_LIBRARY_PATH=/u/richard/torch/install/lib:\$LD_LIBRARY_PATH\n" 
    script += "exec th "+path+"torch-scripts/train-on-mnist.lua \\\n" 
    #script += "--full \\\n" 
    script += "-b 10 \\\n" 
    script += "--rdir "+res_dir+label+" \\\n"
    script += "--top 5 \\\n" 
    script += "--weightDecay 5e-4 \\\n" 
    script += "--momentum 0.9 \\\n" 
    script += "--dropout 0.5 \\\n"
    script += "--learningRate 1e-2 \\\n"
    script += "--train "+data_dir+"train \\\n"
    script += "--test "+data_dir+"test \\\n" 
    script += "--valid "+data_dir+"valid"

    # Write bash script
    with open(script_dir+label+".sh", "w") as f:
        f.write(script)

    # Make bash script executable
    subprocess.call(["chmod", "+x", script_dir+label+".sh"])
    return

    # Make condor script
    condor = ""
    condor += "+Group = \"GRAD\"\n" 
    condor += "+Project = \"OTHER\"\n"
    condor += "+ProjectDescription = \"Breaking privacy-preserving protocols with neural networks\"\n"
    condor += "universe = vanilla\n"
    condor += "Executable = /bin/sh\n"
    condor += "Requirements = InMastodon\n"
    condor += "Notification = Always\n"
    condor += "Notify_user = richardfmcpherson@gmail.com\n"
    condor += "Arguments = "+path+script_dir+label+".sh\n" 
    condor += "Error = "+path+res_dir+label+"/error\n"
    condor += "Output = "+path+res_dir+label+"/output\n"
    condor += "Log = "+path+res_dir+label+"/log\n"
    condor += "Queue 1\n"

    # Write condor script
    with open(script_dir+label+".su", "w") as f:
        f.write(condor)

    # Run condor script
    subprocess.call(["condor_submit", script_dir+label+".su"])


def main():

    # Make directories
    subprocess.call(["mkdir", "-p", script_dir])
    subprocess.call(["mkdir", "-p", res_dir])

    # Normal images
    condor("norm", path+"data/norm/")

    # P3 images
    condor("p3-1", path+"data/p3-1/")
    condor("p3-10", path+"data/p3-10/")
    condor("p3-20", path+"data/p3-20/")

    # Pixelated images
    condor("pix-2", path+"data/pix-2/")
    condor("pix-4", path+"data/pix-4/")
    condor("pix-8", path+"data/pix-8/")
    condor("pix-16", path+"data/pix-16/")

    # Youtube images
    #condor("youtube", path+"data/youtube/")
if __name__ == "__main__":
    main()


