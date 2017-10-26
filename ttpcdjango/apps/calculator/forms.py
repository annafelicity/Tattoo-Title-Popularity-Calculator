from django import forms


class CalculateForm(forms.Form):
    text = forms.CharField(
        label="Query", max_length=255, required=True)