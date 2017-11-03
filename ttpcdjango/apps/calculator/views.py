from django.shortcuts import render
#from django.http import HttpResponseRedirect
from django.urls import reverse
from ipware.ip import get_ip

from .models import QueryLog
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
    calculator_result = None
    template = "calculate_form.html"
    form = CalculateForm()
    if request.method == 'POST':
        form = CalculateForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data.get("text")
            ql = QueryLog(query=query)
            ql.ip_address=get_ip(request)
            ql.save()
            calculator_result = 20
            #20 is fake number; this is where model output will go

            #return HttpResponseRedirect(reverse("thanks"))
    
    context = {"form": form, "calculator_result": calculator_result}
    return render(request, template, context)