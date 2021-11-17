from django.db import models

class Child(models.Model):
    name=models.CharField(max_length=256)
    age=models.IntegerField(null=True)
    persontype=models.CharField(max_length=256)
    child_pic = models.ImageField(upload_to = 'MissingPerson/pic_folder/', default = 'MissingPersonApp/pic_folder/None/no-img.jpg',null=False)
    child_info=models.TextField(null=True)
    contact=models.CharField(null=True,max_length=256)

class Video(models.Model):
    location = models.CharField(max_length=500)
    videofile= models.FileField(upload_to='videos/', null=True, verbose_name="")

    def __str__(self):
        return self.location + ": " + str(self.videofile)

class Person(models.Model):
    phone = models.CharField(default="0",max_length=256)
    name=models.CharField(null=True,max_length=256)
    location=models.CharField(max_length=256)
    child_pic1 = models.ImageField(upload_to = 'MissingPerson/pic_folder/', default = 'MissingPersonApp/pic_folder/None/no-img.jpg',null=False)
    child_pic2 = models.ImageField(upload_to = 'MissingPerson/pic_folder/', default = 'MissingPersonApp/pic_folder/None/no-img.jpg',null=False)
    child_pic3 = models.ImageField(upload_to = 'MissingPerson/pic_folder/', default = 'MissingPersonApp/pic_folder/None/no-img.jpg',null=False)
    child_pic4 = models.ImageField(upload_to = 'MissingPerson/pic_folder/', default = 'MissingPersonApp/pic_folder/None/no-img.jpg',null=False)
    child_pic5 = models.ImageField(upload_to = 'MissingPerson/pic_folder/', default = 'MissingPersonApp/pic_folder/None/no-img.jpg',null=False)
    child_info=models.TextField(null=True)

class Match(models.Model):
    name = models.CharField(max_length=256)
    contact = models.CharField(max_length=15)
    location = models.CharField(max_length=256)
    foundContact = models.CharField(max_length=15)

class HelperPerson(models.Model):
    contact = models.CharField(null=True,max_length=256)