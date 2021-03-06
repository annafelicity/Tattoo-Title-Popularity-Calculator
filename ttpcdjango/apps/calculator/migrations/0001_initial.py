# -*- coding: utf-8 -*-
# Generated by Django 1.11.5 on 2017-10-26 02:27
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='QueryLog',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('query', models.CharField(max_length=255)),
                ('desc', models.TextField(blank=True, help_text='Description for Query.', null=True)),
                ('viewed', models.BooleanField(default=True)),
                ('ip_address', models.GenericIPAddressField(blank=True, null=True)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('modified', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Query',
                'verbose_name_plural': 'Queries',
                'ordering': ['-created'],
            },
        ),
    ]
