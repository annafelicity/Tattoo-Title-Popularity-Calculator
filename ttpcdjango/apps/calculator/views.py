from django.shortcuts import render

# Create your views here.


def hello_world(request):
    context = {
        "name": "Hello World!",
        "colors": ["red", "blue", "green"]}
    template = "hello.html"
    return render(request, template, context)
