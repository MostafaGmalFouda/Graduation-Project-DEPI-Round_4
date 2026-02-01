class Hospital:
    """
    This class represents the hospital and manages the various departments within it.
    It is used to link departments and store basic hospital information.
    """
    
    def __init__(self, name, address):
        """
        Initializes the Hospital attributes.
        :param name: The name of the hospital
        :param address: The physical address of the hospital
        """
        self.name = name          
        self.address = address    
        self.departments = []     

    def add_department(self, department):
        """
        This function is responsible for adding a new department to the hospital.
        :param department: An instance (object) of a Department class
        """
        self.departments.append(department)
        print(f"Department '{department.name}' has been added to {self.name}.")