"""
    Subjects information class
"""

class Subject(object):
    """
        Attributes
        ----------
        id : str
            subject identifier
        
        gender : str
            subject gender : M / F

        age : int
            subject age

        handedness : str
            subject handedness Left / Right

        condition : str
            subject health condition : healthy / a neurodegenartive disease

        Methods
        -------
    """
    
    def __init__(self, id='S', gender='M', age=18, handedness='R', condition='healthy'):
        self.id = id
        self.gender = gender
        self.age = age
        self.handedness = handedness
        self.condition = condition
