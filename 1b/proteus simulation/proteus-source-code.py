import RPi.GPIO as GPIO                           ## Import GPIO Library.
import time                                 ## Import ‘time’ library for a delay.

GPIO.setmode(GPIO.BOARD)                    ## Use BOARD pin numbering.
GPIO.setup(22, GPIO.OUT)                    ## set output.

pwm=GPIO.PWM(22,100)                        ## PWM Frequency
pwm.start(5)

angle1=10
duty1= float(angle1)/10 + 2               ## Angle To Duty cycle  Conversion

angle2=160
duty2= float(angle2)/10 + 2.5

ck=0
while 1:
     pwm.ChangeDutyCycle(duty1)
     time.sleep(0.8)
     pwm.ChangeDutyCycle(duty2)
     time.sleep(0.8)
     ck=ck+1
time.sleep(1)
GPIO.cleanup()
