from django.shortcuts import render
from django import forms

# Create your views here.

def affirmation(request):
	context = {
		"affirmation": "I've got this!",
		"reasons": ["I'm tough", "I don't stop learning", "I'm smart", "I have a thick skin"],
	}
	template = "hello.html"
	return render(request, template, context)

def calculate(request):
    template = "calculate_form.html"
    form = forms.CalculateForm()
    #this is my lesson stuff I'm supposed to do
    context = {"form": form}
    return render(request, template, context)