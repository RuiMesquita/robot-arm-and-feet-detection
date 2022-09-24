import socket
import time


HOST = '10.0.2.15'
PORT = 30002

print("Starting program")

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
time.sleep(0.5)

s.send(("set_digital_out(0, True)"+"\n").encode("utf-8"))
time.sleep(2)

print("Robot starts moving")

s.send(("movej([-0.48,-1.56,-0.99,-2.16,1.57,1.58],a=1.4, v=1.05)"+"\n").encode("utf-8"))
time.sleep(1)

s.send(("movej(p[0.257,0.135,-0.177,1.428,-2.801,-0.001],a=1.4, v=1.05)"+"\n").encode("utf-8"))
time.sleep(1)


s.send(("set_digital_out(0, False)"+"\n").encode("utf-8"))
time.sleep(0.1)

s.close()