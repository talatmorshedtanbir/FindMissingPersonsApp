from django.conf.urls import url
from django.contrib.auth import views as auth_views
from . import views
from rest_framework.routers import DefaultRouter
from phone_verify.api import VerificationViewSet

default_router = DefaultRouter(trailing_slash=False)
default_router.register('phone', VerificationViewSet, basename='phone')

urlpatterns = default_router.urls
app_name = 'child'

urlpatterns=[
    url(r"^addinfo/$", views.HelperView, name='helper'),
    url(r"^searchinfo/$", views.PoliceView, name='police'),
    url(r"^result/$", views.ResultView, name='result'),
    url(r"^person/$", views.PersonView, name='person'),
    url(r"^match/$", views.MatchView, name='match'),
    url(r"^record/$", views.RecordView, name='record'),
    url(r"^personlist/$", views.PersonListView, name='personlist'),
    url(r'^otp/$',views.send_otp,name='otp'),
    url(r'^success/',views.success,name='Success'),
    url(r'^failure/',views.failure,name='Failure')
]
