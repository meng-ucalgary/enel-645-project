Jetson Cards Classification Demo

This code should be run on a Nvidia Jeton board.

Runs a gstreamer pipeline that streams the video over UDP and runs inference on the image
to identify which card is being displayed

The --flip argument is used to flip the video (supports values in steps of 90 degrees)

To run (change 127.0.0.1 to the IP of your PC and the port accoring to your needs):

    python3 label_cards.py --ip 127.0.0.1 --m best_model.h5.tflite --flip 180 --port port_number

To watch the stream (change the port to the port that the label_cards.py script is streaming to):

    gst-launch-1.0.exe -v udpsrc port=port_number ! "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! h264parse ! queue leaky=1 ! decodebin ! videoconvert ! autovideosink sync=false