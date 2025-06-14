# Generated by Django 4.2.20 on 2025-04-10 06:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chatbotApp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ChatHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('session_id', models.CharField(max_length=255)),
                ('role', models.CharField(choices=[('human', 'Human'), ('ai', 'AI')], max_length=10)),
                ('message', models.TextField()),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
