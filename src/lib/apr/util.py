from enum import Enum, auto


class Direcao2D(Enum):
    NORTE = (0, 1)
    SUL = (0, -1)
    OESTE = (-1, 0)
    ESTE = (1, 0)
    NOROESTE = (-1, 1)
    NORDESTE = (1, 1)
    SUDOESTE = (-1, -1)
    SUDESTE = (1, -1)

    def __new__(cls, dx, dy):
        obj = object.__new__(cls)
        obj._value_ = auto()
        obj.__dx = dx
        obj.__dy = dy
        return obj

    @property
    def dx(self):
        return self.__dx

    @property
    def dy(self):
        return self.__dy

    def __mul__(self, magnitude):
        return (self.__dx * magnitude, self.__dy * magnitude)

    def __add__(self, other):
        if not isinstance(other, Direcao2D):
            raise TypeError("Unsupported operand type for +")
        return Direcao2D(self.dx + other.dx, self.dy + other.dy)


class Acao:
    def __init__(self, id):
        self.id = id

    def __eq__(self, outro):
        return self.id == outro.id

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"Acao({self.id})"

    def custo(self):
        raise NotImplementedError


class Movimento2D(Acao):
    def __init__(self, direcao, magnitude=1.0):
        super().__init__(direcao * magnitude)
        self.__direcao = direcao
        self.__magnitude = magnitude

    @property
    def direcao(self):
        return self.__direcao

    @property
    def magnitude(self):
        return self.__magnitude

    @property
    def vetor(self):
        return self.direcao * self.magnitude


class Estado:
    def __init__(self, id):
        self.id = id

    def __eq__(self, outro):
        return self.id == outro.id

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"Estado({self.id})"

    def aplicar(self, acao):
        """
        Gera o estado seguinte a partir da ação (genérico)
        """
        raise NotImplementedError


class Posicao2D(Estado):
    def __init__(self, x, y):
        super().__init__((x, y))
        self.__x = x
        self.__y = y

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    def aplicar(self, movimento):
        dx, dy = movimento.vetor
        return Posicao2D(self.x + dx, self.y + dy)


if __name__ == "__main__":
    import unittest

    class TestDirecao2D(unittest.TestCase):
        def test_direction_properties(self):
            self.assertEqual(Direcao2D.NORTE.dx, 0)
            self.assertEqual(Direcao2D.NORTE.dy, 1)

        def test_direction_multiplication(self):
            result = Direcao2D.NORTE * 2
            self.assertEqual(result, (0, 2))

    class TestMovimento2D(unittest.TestCase):
        def test_movement_properties(self):
            movement = Movimento2D(Direcao2D.NORTE, magnitude=2.0)
            self.assertEqual(movement.direcao, Direcao2D.NORTE)
            self.assertEqual(movement.magnitude, 2.0)

        def test_movement_vector(self):
            movement = Movimento2D(Direcao2D.NORTE, magnitude=2.0)
            self.assertEqual(movement.vetor, (0, 2))

    class TestPosicao2D(unittest.TestCase):
        def test_position_properties(self):
            position = Posicao2D(1, 2)
            self.assertEqual(position.x, 1)
            self.assertEqual(position.y, 2)

        def test_position_application(self):
            position = Posicao2D(1, 2)
            movement = Movimento2D(Direcao2D.NORTE, magnitude=2.0)
            new_position = position.aplicar(movement)
            self.assertEqual(new_position.x, 1)
            self.assertEqual(new_position.y, 4)

        unittest.main()
