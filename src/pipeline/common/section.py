from pydantic import BaseModel, Field


# add a named tuple for the section
class Section(BaseModel):
    x_min: int
    x_max: int
    x_dir: int = Field(default=1)
    y_min: int
    y_max: int
    y_dir: int = Field(default=1)

    def __sub__(self, other: "Section") -> "Section":
        """Subtract another section from this one."""
        return Section(
            x_min=self.x_min - other.x_min,
            x_max=self.x_max - other.x_min,
            x_dir=self.x_dir,
            y_min=self.y_min - other.y_min,
            y_max=self.y_max - other.y_min,
            y_dir=self.y_dir,
        )

    def __add__(self, other: "Section") -> "Section":
        """Add another section to this one."""
        return Section(
            x_min=self.x_min + other.x_min,
            x_max=self.x_max + other.x_min,
            x_dir=self.x_dir,
            y_min=self.y_min + other.y_min,
            y_max=self.y_max + other.y_min,
            y_dir=self.y_dir,
        )

    @classmethod
    def from_str(cls, label: str) -> "Section":
        """There is a header convention in fits files that defines a data range"""
        x_min, x_max, y_min, y_max = [int(i) for i in label[1:-1].replace(":", ",").split(",")]
        x_dir, y_dir = 1, 1
        if x_max < x_min:
            x_dir = -1
            x_min, x_max = x_max, x_min
        if y_max < y_min:
            y_dir = -1
            y_min, y_max = y_max, y_min
        return cls(
            x_min=x_min - 1,
            x_max=x_max,
            x_dir=x_dir,
            y_min=y_min - 1,
            y_max=y_max,
            y_dir=y_dir,
        )
