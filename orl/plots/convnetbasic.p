#set terminal dumb 225 50
set autoscale                        # scale axes automatically
unset log                            # say no to log
set multiplot layout 2,2 rowsfirst scale 1,1

set label 1 "Training Loss" at graph 0.05,0.7
plot "results/20/convnetbasic/training" using 1:2 title '1' with linespoints, "results/20/convnetbasic2/training" using 1:2 title '2' with linespoints,  "results/20/convnetbasic3/training" using 1:2 title '3' with linespoints

set label 1 "Training Accuracy" at graph 0.05,0.7
plot "results/20/convnetbasic/training" using 1:3 title '1' with linespoints, "results/20/convnetbasic2/training" using 1:3 title '2' with linespoints, "results/20/convnetbasic3/training" using 1:3 title '3' with linespoints

set label 1 "Testing Loss" at graph 0.05,0.7
plot "results/20/convnetbasic/testing" using 1:2 title '1' with linespoints, "results/20/convnetbasic2/testing" using 1:2 title '2' with linespoints, "results/20/convnetbasic3/testing" using 1:2 title '3' with linespoints

set label 1 "Testing Accuracy" at graph 0.05,0.7
plot "results/20/convnetbasic/testing" using 1:3 title '1' with linespoints, "results/20/convnetbasic2/testing" using 1:3 title '2' with linespoints, "results/20/convnetbasic3/testing" using 1:3 title '3' with linespoints

unset multiplot
