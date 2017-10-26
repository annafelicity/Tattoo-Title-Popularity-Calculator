from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse

from .forms import CalculateForm

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
    form = CalculateForm()
    if request.method == 'POST':
        form = CalculateForm(request.POST)
        if form.is_valid():
            return HttpResponseRedirect(reverse("thanks"))
    
    context = {"form": form}
    return render(request, template, context)