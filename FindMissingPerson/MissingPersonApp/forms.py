from django import forms
from . models import Child,Video,Person

class ChildForm(forms.ModelForm):
    class Meta:
        model = Child
        fields = ('name','age','contact','child_pic', 'persontype')

class VideoForm(forms.ModelForm):
    class Meta:
        model= Video
        fields= ["location", "videofile"]

        
class PersonForm(forms.ModelForm):
    class Meta:
        model= Person
        fields= ["phone","name","location","child_pic1","child_pic2","child_pic3","child_pic4","child_pic5","child_info"]

class MyForm(forms.Form):
	mobile_no=forms.CharField(max_length=20)

class OTPForm(forms.Form):
	otp=forms.IntegerField()