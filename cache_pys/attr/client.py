class Client:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def __str__(self):
        return f"Client(name={self.name}, age={self.age})"
    def __repr__(self):
        return f"Client(name={self.name}, age={self.age})"
    def __eq__(self, other):
        if not isinstance(other, Client):
            return False
        return self.name == other.name and self.age == other.age
    def __hash__(self):
        return hash((self.name, self.age))
    def __lt__(self, other):
        if not isinstance(other, Client):
            return NotImplemented
        return self.age < other.age
    def __le__(self, other):
        if not isinstance(other, Client):
            return NotImplemented
        return self.age <= other.age
    def __gt__(self, other):
        if not isinstance(other, Client):
            return NotImplemented
        return self.age > other.age
    def __ge__(self, other):
        if not isinstance(other, Client):
            return NotImplemented
        return self.age >= other.age
    def __ne__(self, other):
        if not isinstance(other, Client):
            return NotImplemented
        return self.name != other.name or self.age != other.age
    def __add__(self, other):
        if not isinstance(other, Client):
            return NotImplemented
        return Client(self.name + other.name, self.age + other.age)
    def __sub__(self, other):
        if not isinstance(other, Client):
            return NotImplemented
        return Client(self.name.replace(other.name, ""), self.age - other.age)
    def __mul__(self, other):
        if not isinstance(other, Client):
            return NotImplemented
        return Client(self.name * other.age, self.age * other.age)

    def help():
        return """
        Client class:
        - __init__(self, name: str, age: int): Initializes a new Client instance with a name and age.
        - __str__(self): Returns a string representation of the Client instance.
        - __repr__(self): Returns a string representation of the Client instance for debugging.
        - __eq__(self, other): Compares two Client instances for equality.
        - __hash__(self): Returns a hash value for the Client instance.
        - __lt__(self, other): Compares two Client instances for less than.
        - __le__(self, other): Compares two Client instances for less than or equal to.
        - __gt__(self, other): Compares two Client instances for greater than.
        - __ge__(self, other): Compares two Client instances for greater than or equal to.
        - __ne__(self, other): Compares two Client instances for not equal.
        - __add__(self, other): Adds two Client instances together.
        - __sub__(self, other): Subtracts one Client instance from another.
        - __mul__(self, other): Multiplies two Client instances together.
        """


def help():
    return """
    Client class:
    """
