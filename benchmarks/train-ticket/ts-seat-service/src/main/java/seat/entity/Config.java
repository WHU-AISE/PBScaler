package seat.entity;

import lombok.Data;

/**
 * @author fdse
 */
@Data
public class Config {

    private String name;

    private String value;

    private String description;

    public Config() {
    }

    public Config(String name, String value, String description) {
        this.name = name;
        this.value = value;
        this.description = description;
    }

}
