from django.shortcuts import render

# Create your views here.

def affirmation(request):
	context = {
		"affirmation": "I've got this!",
		"reasons": ["I'm tough", "I don't stop learning", "I'm smart", "I have a thick skin"],
	}
	template = "hello.html"
	return render(request, template, context)