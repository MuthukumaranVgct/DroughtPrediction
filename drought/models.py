from django.db import models

# Create your models here.

class datasetPrec(models.Model):
    place=models.CharField(max_length=25)
    year=models.IntegerField()
    jan=models.FloatField()
    feb=models.FloatField()
    mar=models.FloatField()
    apr=models.FloatField()
    may=models.FloatField()
    jun=models.FloatField()
    july=models.FloatField()
    aug=models.FloatField()
    sep=models.FloatField()
    octb=models.FloatField()
    nov=models.FloatField()
    dec=models.FloatField()

class datasetTemp(models.Model):
    place=models.CharField(max_length=25)
    year=models.IntegerField()
    jan=models.FloatField()
    feb=models.FloatField()
    mar=models.FloatField()
    apr=models.FloatField()
    may=models.FloatField()
    jun=models.FloatField()
    july=models.FloatField()
    aug=models.FloatField()
    sep=models.FloatField()
    octb=models.FloatField()
    nov=models.FloatField()
    dec=models.FloatField()

class datasetPET(models.Model):
    place=models.CharField(max_length=25)
    year=models.IntegerField()
    jan=models.FloatField()
    feb=models.FloatField()
    mar=models.FloatField()
    apr=models.FloatField()
    may=models.FloatField()
    jun=models.FloatField()
    july=models.FloatField()
    aug=models.FloatField()
    sep=models.FloatField()
    octb=models.FloatField()
    nov=models.FloatField()
    dec=models.FloatField()

class latlon(models.Model):
    place=models.CharField(max_length=25)
    lat=models.FloatField()
    lon=models.FloatField()

