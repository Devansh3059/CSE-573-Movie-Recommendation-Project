from flask_wtf import FlaskForm
from wtforms import SubmitField,StringField
from wtforms.validators import DataRequired

class MovieForm(FlaskForm):
    moviename=StringField('Enter Movie Name',validators=[DataRequired()])
    submit=SubmitField('Search')

class UserForm(FlaskForm):
    moviename=StringField('Enter User Id',validators=[DataRequired()])
    submit=SubmitField('Search')

