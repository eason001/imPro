from django.shortcuts import render



def index(request):
    context = {'hello': 'Hello Yi'}
    return render(request, 'impro/index.html', context)
