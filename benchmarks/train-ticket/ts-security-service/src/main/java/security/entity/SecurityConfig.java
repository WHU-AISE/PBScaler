package security.entity;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;
import org.springframework.data.mongodb.core.mapping.Document;
import java.util.UUID;

/**
 * @author fdse
 */
@Data
@Document(collection = "security_config")
@JsonIgnoreProperties(ignoreUnknown = true)
public class SecurityConfig {

    private UUID id;

    private String name;

    private String value;

    private String description;

    public SecurityConfig() {
        //Default Constructor
    }

}
