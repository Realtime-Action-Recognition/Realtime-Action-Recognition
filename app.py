# ==============================================================================================
# Title:   test_ucf101.py
# Contact: realtimeactionrecognition@gmail.com
# ==============================================================================================

import sys
import cv2
import json
from realtime_action_recognition import ActionSystem
from flask import Flask, render_template, Response, send_file, request, jsonify

import action_steps as handwash_steps

step_images = [
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done"]

app = Flask(__name__)

step_information = ""

handwash_steps = handwash_steps.ActionSteps()

@app.route('/')
def index():
    return render_template('index.html', step_image=step_images)

def gen(live_stream):
    global step_information
    while True:
        response_from_system = live_stream.get_frame()
        if len(response_from_system) == 2:
          step_information = handwash_steps.incorrect_step_order(response_from_system[1])
          handwash_steps.add_step(response_from_system[1])

        frame = response_from_system[0]
        yield (b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/step_message')
def step_message():
  return handwash_steps.get_next_step()

@app.route('/step_info')
def step_info():
  return step_information

@app.route('/step_gif')
def step_gif():
  step_number = handwash_steps.get_step_number(handwash_steps.get_next_step())
  gif_name = "STEP "+str(step_number)+".gif"
  return "/static/"+gif_name  

@app.route('/step_1_checkbox')
def step_1_checkbox():
  if handwash_steps.steps["STEP 1"] == 'complete':
    filename = "/static/done.png"
  else:
    filename = "/static/not_done.png"
  return filename

@app.route('/step_2_left_checkbox')
def step_2_left_checkbox():
  if handwash_steps.steps["STEP 2 LEFT"] == 'complete':
    filename = "/static/done.png"
  else:
    filename = "/static/not_done.png"
  return filename

@app.route('/step_2_right_checkbox')
def step_2_right_checkbox():
  if handwash_steps.steps["STEP 2 RIGHT"] == 'complete':
    filename = "/static/done.png"
  else:
    filename = "/static/not_done.png"
  return filename

@app.route('/step_3_checkbox')
def step_3_checkbox():
  if handwash_steps.steps["STEP 3"] == 'complete':
    filename = "/static/done.png"
  else:
    filename = "/static/not_done.png"
  return filename

@app.route('/step_4_left_checkbox')
def step_4_left_checkbox():
  if handwash_steps.steps["STEP 4 LEFT"] == 'complete':
    filename = "/static/done.png"
  else:
    filename = "/static/not_done.png"
  return filename

@app.route('/step_4_right_checkbox')
def step_4_right_checkbox():
  if handwash_steps.steps["STEP 4 RIGHT"] == 'complete':
    filename = "/static/done.png"
  else:
    filename = "/static/not_done.png"
  return filename
      
@app.route('/step_5_left_checkbox')
def step_5_left_checkbox():
  if handwash_steps.steps["STEP 5 LEFT"] == 'complete':
    filename = "/static/done.png"
  else:
    filename = "/static/not_done.png"
  return filename

@app.route('/step_5_right_checkbox')
def step_5_right_checkbox():
  if handwash_steps.steps["STEP 5 RIGHT"] == 'complete':
    filename = "/static/done.png"
  else:
    filename = "/static/not_done.png"
  return filename

@app.route('/step_6_left_checkbox')
def step_6_left_checkbox():
  if handwash_steps.steps["STEP 6 LEFT"] == 'complete':
    filename = "/static/done.png"
  else:
    filename = "/static/not_done.png"
  return filename

@app.route('/step_6_right_checkbox')
def step_6_right_checkbox():
  if handwash_steps.steps["STEP 6 RIGHT"] == 'complete':
    filename = "/static/done.png"
  else:
    filename = "/static/not_done.png"
  return filename

@app.route('/step_7_left_checkbox')
def step_7_left_checkbox():
  if handwash_steps.steps["STEP 7 LEFT"] == 'complete':
    filename = "/static/done.png"
  else:
    filename = "/static/not_done.png"
  return filename

@app.route('/step_7_right_checkbox')
def step_7_right_checkbox():
  if handwash_steps.steps["STEP 7 RIGHT"] == 'complete':
    filename = "/static/done.png"
  else:
    filename = "/static/not_done.png"
  return filename

@app.route('/video_feed')
def video_feed():
    return Response(gen(HandwashSystem()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def handler(signal, frame):
  print('CTRL-C pressed!')
  sys.exit(0)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port = 8001, use_reloader=False)