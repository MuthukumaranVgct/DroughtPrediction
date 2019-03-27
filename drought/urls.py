from django.conf.urls import url
from drought.views import index
urlpatterns = [
    url('',index,name='index'),
]
