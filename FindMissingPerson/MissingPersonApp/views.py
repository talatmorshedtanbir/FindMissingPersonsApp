from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.urls import reverse_lazy, reverse
from django.views.generic import CreateView, TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from . import forms
from .models import Video, Child, Person, Match, HelperPerson
from .forms import VideoForm, ChildForm, PersonForm, MyForm, OTPForm
import sys
from . import npwriter, npwriter2, npwriter3
from sklearn.neighbors import KNeighborsClassifier
import argparse
import cv2
import numpy as np
import pandas as pd
import os
import face_recognition
import glob
from PIL import Image
from matplotlib import cm
from matplotlib import pyplot as plt
import imutils
import re
import pickle
from sklearn import svm
from rest_framework.routers import DefaultRouter
from phone_verify.api import VerificationViewSet
from twilio.rest import Client
from sendsms.message import SmsMessage
import random
import os

# def PoliceView(request):
#     if request.method == "POST":
#         form = ChildForm(request.POST, request.FILES)
#         if form.is_valid():
#             form.save()
#             data = pd.read_csv(npwriter.f_name).values
#             X, Y = data[:, 1:-1], data[:, -1]
#             model = KNeighborsClassifier(n_neighbors = 2, algorithm='ball_tree', weights='distance')

#             # fdtraining of model
#             model.fit(X, Y)
#             # image1 = face_recognition.load_image_file('media/'+str(Child.objects.last().child_pic))
#             # list_of_face_encodings1 = np.array(face_recognition.face_encodings(image1))
#             # images = []
#             # for img in glob.glob("data/*"):
#             #     print(img)
#             #     image2 = face_recognition.load_image_file(img)
#             #     list_of_face_encodings2 = np.array(face_recognition.face_encodings(image2))
#             #     if len(list_of_face_encodings2) == 0:
#             #         continue
#             #     results = face_recognition.compare_faces(list_of_face_encodings1, list_of_face_encodings2)
#             #     print(results[0])
#             #     if results[0] == True:
#             #         BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#             #         url=os.path.join(BASE_DIR, img)
#             #         sentence = img
#             #         word = "_"
#             #         s=-1
#             #         e=-1
#             #         for match in re.finditer(word, sentence):
#             #             if e==-1:
#             #                 e=match.end()
#             #             else:
#             #                 s=match.start()
#             facedata = "media/haarcascade_frontalface_default.xml"
#             cascade = cv2.CascadeClassifier(facedata)
#             image = cv2.imread('media/'+str(Child.objects.last().child_pic))
#             gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#             gray= cv2.equalizeHist(gray)
#             faces = cascade.detectMultiScale(gray,1.2,5)

#             for f in faces:
#                 x, y, w, h = [ v for v in f ]
#                 sub_face = image[y:y+h, x:x+w]

#             X_test = []
#             gray_face = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
#             gray_face = cv2.resize(gray_face, (100, 100))
#             X_test.append(gray_face.reshape(-1))
#             #response = model.predict(np.array(X_test))
#             dist, inds =model.kneighbors(X_test)
#             print(dist[0][0])
#             narr=np.array(dist[0])
#             if(np.any(narr<3500)):
#                 predicted_label = model.predict(X_test)
#                 return render(request, 'result.html',{'text':predicted_label,'found':'Person Found'})
#             else:
#                 return render(request, 'result.html',{'text':'Person not found','found':'Person not Found'})
#         else:
#             print("Form not valid")
#     else:
#         form = ChildForm()
#     return render(request, 'policeinfo.html', {'form': form})


def HelperView(request):
    if request.method == "POST":
        print(request.FILES['videofile'])
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            data = pd.read_csv(npwriter3.f_name).values
            X_train, Y_train = data[:, 1:-1], data[:, -1]
            model = KNeighborsClassifier(
                n_neighbors=1, algorithm='ball_tree', weights='distance')
            # fdtraining of model
            model.fit(X_train, Y_train)

            facedata = "media/haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(facedata)

            FPS = 10
            cap = cv2.VideoCapture(
                'media/'+str(Video.objects.last().videofile))
            cap.set(cv2.CAP_PROP_FPS, FPS)
            j = cap.get(cv2.CAP_PROP_FPS)
            print(j)
            try:
                if not os.path.exists('data'):
                    os.makedirs('data')
            except OSError:
                print('Error: Creating directory of data')
            currentFrame = 0
            while(True):
                ret, frame = cap.read()
                if not ret:
                    break
                small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
                rgb_small_frame = small_frame[:, :, ::-1]
                face_locations = face_recognition.face_locations(
                    rgb_small_frame)
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations)

                for face_encoding in face_encodings:
                    testf = np.array(face_encoding)
                    testf = testf.reshape(1, -1)

                    dist, inds = model.kneighbors(testf)
                    narr = np.array(dist[0])
                    labels = []
                    matcheddata = []
                    if(np.any(narr < 0.4)):
                        predicted_label = model.predict(testf)
                        labels = predicted_label[0].split(",")
                        matcheddata.append((labels[0], labels[1]))
                        file_name = "VideoData/img" + "_" + \
                            labels[0]+"_"+labels[1] + "_" + ".jpg"
                        cv2.imwrite(file_name, rgb_small_frame)
                        print("Found a missing person")
                        print("Name: "+labels[1]+"  Contact:"+labels[0])
                        account_sid = 'AC60dbe35fc5f3b06e879f2f3613fc3b38'
                        auth_token = '976486b7e3e5d2286327b7147954b6e6'
                        client = Client(account_sid, auth_token)
                        loc = Video.objects.last().location
                        client.messages.create(to='+8801817316436', from_='+14695303875', body="A person was found. Name is " +
                                               labels[1]+", Contact no is "+labels[0]+", Found location is "+loc)
                        return render(request, 'vidmatch.html', {'data': matcheddata})
                currentFrame += 1
            cap.release()
            cv2.destroyAllWindows()
            return redirect('home')
        else:
            print("Form not valid")
    else:
        form = VideoForm()
    return render(request, 'helper.html', {'form': form})


def ResultView(request):
    return render(request, "result.html")


def PersonView2(request):
    if request.method == "POST":
        form = PersonForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            facedata = "media/haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(facedata)
            image = cv2.imread('media/'+str(Person.objects.last().child_pic1))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = cascade.detectMultiScale(gray, 1.2, 5)
            i = 0
            for f in faces:
                x, y, w, h = f
                sub_face = image[y:y+h, x:x+w]
                file_name = "FoundData/img" + "_" + Person.objects.last().location + "_" + \
                    str(i)+str(Person.objects.last().phone) + ".jpg"
                cv2.imwrite(file_name, sub_face)
                i += 1

            f_list = []
            gray_face = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (100, 100))
            print(np.shape(gray_face))
            f_list.append(gray_face.reshape(-1))
            npwriter.write(str(Person.objects.last().phone)+" , " +
                           Person.objects.last().location, np.array(f_list))
            return redirect('home')
        else:
            print("Form Not Valid")
    else:
        form = PersonForm()
    return render(request, 'person.html', {'form': form})


def PersonView1(request):
    if request.method == "POST":
        form = PersonForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            images = []
            empty_file = "MissingPersonApp/pic_folder/None/no-img.jpg"

            if(str(Person.objects.last().child_pic1) != empty_file):
                images.append(face_recognition.load_image_file(
                    'media/'+str(Person.objects.last().child_pic1)))
            if(str(Person.objects.last().child_pic2) != empty_file):
                images.append(face_recognition.load_image_file(
                    'media/'+str(Person.objects.last().child_pic2)))
            if(str(Person.objects.last().child_pic2) != empty_file):
                images.append(face_recognition.load_image_file(
                    'media/'+str(Person.objects.last().child_pic2)))

            i = 0
            for img in images:
                file_name = "FoundData/img" + "_" + Person.objects.last().location + \
                    str(i) + "_" + ".jpg"
                cv2.imwrite(file_name, img)
                i += 1
                list_of_face_encodings = np.array(
                    face_recognition.face_encodings(img))
                npwriter2.write(str(Person.objects.last(
                ).phone)+" , " + Person.objects.last().location, np.array(list_of_face_encodings))
            return redirect('home')
        else:
            print("Form Not Valid")
    else:
        form = PersonForm()
    return render(request, 'person.html', {'form': form})


def PersonView(request):
    if request.method == "POST":
        form = PersonForm(request.POST, request.FILES)
        contactno = HelperPerson.objects.last().contact
        print(contactno)
        if form.is_valid():
            form.cleaned_data['phone'] = contactno
            form.save()
            images = []
            empty_file = "MissingPersonApp/pic_folder/None/no-img.jpg"

            if(str(Person.objects.last().child_pic1) != empty_file):
                images.append(face_recognition.load_image_file(
                    'media/'+str(Person.objects.last().child_pic1)))
            if(str(Person.objects.last().child_pic2) != empty_file):
                images.append(face_recognition.load_image_file(
                    'media/'+str(Person.objects.last().child_pic2)))
            if(str(Person.objects.last().child_pic3) != empty_file):
                images.append(face_recognition.load_image_file(
                    'media/'+str(Person.objects.last().child_pic3)))
            if(str(Person.objects.last().child_pic4) != empty_file):
                images.append(face_recognition.load_image_file(
                    'media/'+str(Person.objects.last().child_pic4)))
            if(str(Person.objects.last().child_pic5) != empty_file):
                images.append(face_recognition.load_image_file(
                    'media/'+str(Person.objects.last().child_pic5)))
            i = 0
            emp = []
            facedata = "media/haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(facedata)
            for img in images:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                face = cascade.detectMultiScale(gray, 1.1, 4)
                for f in face:
                    x, y, w, h = f
                sub_face = img[y:y + h, x:x + w]
                file_name = "FoundData/img" + "_" + Person.objects.last().location + \
                    str(i) + "_" + ".jpg"
                cv2.imwrite(file_name, sub_face)
                list_of_face_encodings = np.array(
                    face_recognition.face_encodings(sub_face))
                if(list_of_face_encodings != emp):
                    npwriter2.write(str(Person.objects.last().phone).replace(
                        '+', '')+" , " + Person.objects.last().location, np.array(list_of_face_encodings))
                i += 1
            return redirect('home')
        else:
            print("Form Not Valid")
    else:
        form = PersonForm()
    return render(request, 'person.html', {'form': form})


def PoliceView(request):
    if request.method == "POST":
        form = ChildForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            data = pd.read_csv(npwriter2.f_name).values
            X_train, Y_train = data[:, 1:-1], data[:, -1]
            model = KNeighborsClassifier(
                n_neighbors=2, algorithm='ball_tree', weights='distance')
            # fdtraining of model
            model.fit(X_train, Y_train)

            image = face_recognition.load_image_file(
                'media/'+str(Child.objects.last().child_pic))
            file_name = "MissingData/img" + "_" + Child.objects.last().name + "_" + ".jpg"
            cv2.imwrite(file_name, image)
            list_of_face_encodings = np.array(
                face_recognition.face_encodings(image))
            dist, inds = model.kneighbors(list_of_face_encodings)
            print(dist[0])
            narr = np.array(dist[0])
            resText = []
            labels = []
            if(np.any(narr < 0.4)):
                predicted_label = model.predict(list_of_face_encodings)

                labels = predicted_label[0].split(",")

                match = Match(name=Child.objects.last().name, contact=str(
                    Child.objects.last().contact), location=labels[1], foundContact=labels[0])
                match.save()
                resText.append(Child.objects.last().name)
                resText.append(str(Child.objects.last().contact))
                resText.append(labels[0])
                resText.append(labels[1])
                return render(request, 'result.html', {'text': resText, 'found': 'Found'})
            else:
                npwriter3.write(str(Child.objects.last().contact)+" , " +
                                Child.objects.last().name, np.array(list_of_face_encodings))
                return render(request, 'result.html', {'text': 'Person not found', 'found': 'Not Found'})
        else:
            print("Form not valid")
    else:
        form = ChildForm()
    return render(request, 'policeinfo.html', {'form': form})


def MatchView(request):
    data = pd.read_csv(npwriter2.f_name).values
    X_train, Y_train = data[:, 1:-1], data[:, -1]
    model = KNeighborsClassifier(
        n_neighbors=1, algorithm='ball_tree', weights='distance')
    # fdtraining of model
    model.fit(X_train, Y_train)

    readCSV = pd.read_csv(npwriter3.f_name)
    data = readCSV.values
    X, Y = data[:, 1:-1], data[:, -1]

    predictedValues = []
    matcheddata = []
    i = 0
    for x in X:
        x_new = np.array(x).reshape(1, -1)
        dist, inds = model.kneighbors(x_new)
        narr = np.array(dist[0])
        if(np.any(narr <= 0.4)):
            predicted_label = model.predict(x_new)
            label = predicted_label[0]
            print(label)
            vals = []
            vals = label.split(",")
            labs = []
            labs = Y[i].split(",")
            print("label is" + str(vals[1]))
            match = Match(name=labs[1], contact=labs[0],
                          location=vals[0], foundContact=vals[1])
            match.save()
            readCSV = readCSV.drop(i, axis=0)
            readCSV.to_csv(npwriter3.f_name, index=False)
            matcheddata.append((labs[1], labs[0], vals[1], vals[0]))
        i += 1
    return render(request, 'match.html', {'data': matcheddata})


def PoliceViewSVM(request):
    if request.method == "POST":
        form = ChildForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            data = pd.read_csv(npwriter2.f_name).values
            X, Y = data[:, 1:-1], data[:, -1]

            model = svm.SVC(kernel='linear', probability=True)
            model.fit(X, Y)

            image = face_recognition.load_image_file(
                'media/'+str(Child.objects.last().child_pic))
            list_of_face_encodings = np.array(
                face_recognition.face_encodings(image))
            resText = []
            probal = model.predict_proba(list_of_face_encodings)

            if(np.any(probal[0] >= 0.7)):
                predicted_label = model.predict(list_of_face_encodings)
                print("pred label" + predicted_label)
                labels = predicted_label[0].split(",")
                print(labels[0])
                resText.append(Child.objects.last().name)
                resText.append(str(Child.objects.last().contact))
                resText.append(labels[0])
                resText.append(labels[1])

            return render(request, 'result.html', {'text': resText, 'found': 'Found'})
        else:
            print("Form not valid")
    else:
        form = ChildForm()
    return render(request, 'policeinfo.html', {'form': form})


def RecordView(request):
    records = Match.objects.all()
    return render(request, 'record.html', {'data': records})


def PersonListView(request):
    plist = Child.objects.all()
    return render(request, 'personlist.html', {'data': plist})


def generate_otp(mobile_no):
    otp = random.randint(1000, 9999)
    account_sid = ''
    auth_token = ''
    client = Client(account_sid, auth_token)
    client.messages.create(to=mobile_no, from_='+14695303875',
                           body='Your One Time Password is '+str(otp))
    f = open('otp.txt', 'w')
    f.write(str(otp))
    f.close()


def send_otp(request):

    form = MyForm(request.POST)
    otp_form = OTPForm(request.POST)
    otp = 0
    mob = 0
    if form.is_valid():
        cd = form.cleaned_data
        mobile_no = cd.get('mobile_no')
        if os.path.getsize('otp.txt') == 0:
            helper = HelperPerson(contact=mobile_no)
            helper.save()
            generate_otp(mobile_no)

    elif otp_form.is_valid():
        f = open('otp.txt', 'r')
        otp = int(f.read())
        f.close()
        print(otp)
        open('otp.txt', 'w').close()
        cd2 = otp_form.cleaned_data
        entered_otp = cd2.get('otp')
        if otp == entered_otp:
            print(mob)
            return render(request, 'success.html')
        else:
            return render(request, 'failure.html')

    return render(request, 'otp.html', {'form': form, 'otp_form': otp_form})


def success(request):
    return render(request, 'success.html')


def failure(request):
    return render(request, 'failure.html')
