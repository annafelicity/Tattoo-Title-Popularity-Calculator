from django import forms


class CalculateForm(forms.Form):
    text = forms.CharField(
        label="Enter your title here", max_length=255, required=True)