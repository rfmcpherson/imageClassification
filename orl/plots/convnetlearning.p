#set terminal dumb 225 50
set autoscale                        # scale axes automatically
unset log                            # say no to log
set multiplot layout 2,2 rowsfirst scale 1,1

set label 1 "Training Loss" at graph 0.05,0.7
plot "results/20/convnet/training" using 1:2 title '0.001' with linespoints, "results/20/convnetr0d005/training" using 1:2 title '0.005' with linespoints,  "results/20/convnetr0d01/training" using 1:2 title '0.01' with linespoints, "results/20/convnetr0d05/training" using 1:2 title '0.05' with linespoints 

set label 1 "Training Accuracy" at graph 0.05,0.7
plot "results/20/convnet/training" using 1:3 title '0.001' with linespoints, "results/20/convnetr0d005/training" using 1:3 title '0.005' with linespoints, "results/20/convnetr0d01/training" using 1:3 title '0.01' with linespoints, "results/20/convnetr0d05/training" using 1:3 title '0.05' with linespoints 

set label 1 "Testing Loss" at graph 0.05,0.7
plot "results/20/convnet/testing" using 1:2 title '0.001' with linespoints, "results/20/convnetr0d005/testing" using 1:2 title '0.005' with linespoints, "results/20/convnetr0d01/testing" using 1:2 title '0.01' with linespoints, "results/20/convnetr0d05/testing" using 1:2 title '0.05' with linespoints 

set label 1 "Testing Accuracy" at graph 0.05,0.7
plot "results/20/convnet/testing" using 1:3 title '0.001' with linespoints, "results/20/convnetr0d005/testing" using 1:3 title '0.005' with linespoints, "results/20/convnetr0d01/testing" using 1:3 title '0.01' with linespoints, "results/20/convnetr0d05/testing" using 1:3 title '0.05' with linespoints 

unset multiplot
