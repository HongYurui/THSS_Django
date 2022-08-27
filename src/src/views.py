import os
import django
import datetime
import re
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "project_name.settings")
django.setup()
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.core.paginator import Paginator
from app import models

import time

def home(request):
    context = {}
    context['home'] = "https://github.com/HongYurui"
    context['contact'] = "hongyr20@mails.tsinghua.edu.cn"
    return render(request, 'home.html', context)

def train(request):
    return render(request, 'train.html')
    
def submit(request):
    model_name = request.GET.get('model_name')
    uploader = request.GET.get('uploader')
    learning_rate = request.GET.get('learning_rate')
    batch_size = request.GET.get('batch_size')
    epochs = request.GET.get('epochs')
    context: dict = {"model_name"           : model_name,
                     "uploader"             : uploader,
                     "learning_rate"        : learning_rate,
                     "batch_size"           : batch_size,
                     "epochs"               : epochs,
                     "error_learning_rate"  : "",
                     "error_batch_size"     : "",
                     "error_epochs"         : ""}
    
    # record radio checked value
    if request.GET.get('lenet_structure') == "LeNet":
        context["lenet_structure"] = "LeNet"
        context['checked_lenet'] = "checked"
    else:
        context["lenet_structure"] = "LeNetP"
        context['checked_lenetp'] = "checked"
    if request.GET.get('optimizer') == "SGD":
        context["optimizer"] = "SGD"
        context['checked_sgd'] = "checked"
    else:
        context["optimizer"] = "Adam"
        context['checked_adam'] = "checked"
    
    # check input format
    for name in ['model_name', 'uploader']:
        if not re.match('^\\w+$', context[name]):
            context['error_' + name] = "Only letters, numbers and underscores are allowed"
            break
    try:
        learning_rate = float(learning_rate)
        assert learning_rate > 0 and learning_rate < 1
    except (ValueError, AssertionError):
        context['error_learning_rate'] = "Learning rate must be a float between 0 and 1!"
    try:
        batch_size = int(batch_size)
        assert batch_size > 0
    except (ValueError, AssertionError):
        context['error_batch_size'] = "Batch size must be an integer greater than 0!"
    try:
        epochs = int(epochs)
        assert epochs > 0
    except (ValueError, AssertionError):
        context['error_epochs'] = "Epochs must be an integer greater than 0!"
        
    saved_data:dict = {}
    for key in context.keys():
        if re.match("^error_\\w*", key):
            if context[key] != "":
                return render(request, 'train.html', context)
        else:
            saved_data[key] = context[key]
                
    # save to database
    timestamp = datetime.datetime.now().replace(microsecond=0)
    models.Record.objects.create(model_name=model_name, uploader=uploader, created_at=timestamp, updated_at=timestamp)
    saved_data['time'] = timestamp
    
    # begin training
    os.popen("python3 src/launcher.py --args \""+str(saved_data)+"\"")
    
    # redirect to records page
    return redirect('/records?model_name=&uploader=&page_num=1')


def model(request, created_at):
    record = models.Record.objects.get(created_at=datetime.datetime.strptime(created_at, '%Y%m%d%H%M%S'))
    context = {'record': record}
    if record.created_at == record.updated_at:
        context['timespan'] = datetime.datetime.now().replace(microsecond=0) - record.created_at
    else:
        context['timespan'] = record.updated_at-record.created_at
    dir_name = "static/models/" + record.created_at.strftime("%Y%m%d%H%M%S")
    context['log'] = open(dir_name + "/output.log", "r").read()
    return render(request, 'model.html', context)
    
def records(request, page_num=1, page_size=2, paginator_width=2):
    
    objects = models.Record.objects
    model_name = request.GET.get('model_name')
    uploader = request.GET.get('uploader')
    page_num = request.GET.get('page_num') or page_num
    filtered_records = objects.order_by('-created_at')
    
    context = {
                'paginator_width': paginator_width,
                'filter_model_name': model_name,
                'filter_uploader': uploader,
              }
    
    # search pattern
    if request.GET.get('search') == "exact":
        context['search'] = "exact"
        context['checked_exact'] = "checked"
    else:
        context['search'] = "fuzzy"
        context['checked_fuzzy'] = "checked"
    
    # filter logic
    if context['search'] == "exact":
        if model_name:
            filtered_records = filtered_records.filter(model_name=model_name)
        if uploader:
            filtered_records = filtered_records.filter(uploader=uploader)
    else:
        filtered_records = filtered_records.filter(model_name__icontains=model_name)
        filtered_records = filtered_records.filter(uploader__icontains=uploader)
    
    # paginator
    context['page_count'] = (objects.count() - 1) // page_size + 1
    context['paginator'] = Paginator(filtered_records, page_size)
    context['page'] = context['paginator'].page(page_num)
    
    return render(request, 'records.html', context)