from django.shortcuts import render
from Filtering_Module.WordFilteringWithBERT import filtering_bert
from Filtering_Module.vector import *


# Create your views here.
def home(request):
    return render(request, 'home.html')


def result(request):
    input_value = request.GET['input']

    filtering_result = convert(input_value)

    if not filtering_result.startswith('*'):
        filtering_result = filtering_bert(input_value)
    return render(request, 'result.html', {'original': input_value, 'result': filtering_result})
