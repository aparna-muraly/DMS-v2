import Jetson.GPIO as GPIO
import time

# Set the GPIO mode
GPIO.setmode(GPIO.BOARD)

# Set pin 29 (GPIO1) as input (digital input)
GPIO.setup(29, GPIO.IN)

# Infinite loop to read the sensor data
try:
    while True:
        if GPIO.input(29) == GPIO.LOW:
            print("Alcohol detected!")
        else:
            print("No alcohol detected")
        time.sleep(1)

except KeyboardInterrupt:
    GPIO.cleanup()

