package cancel.entity;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

import java.util.UUID;

/**
 * @author fdse
 */
@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class Account {

    private UUID id;

    private String password;

    private int gender;

    private String name;

    private int documentType;

    private String documentNum;

    private String email;

    public Account(){
        gender = Gender.OTHER.getCode();
        password = "defaultPassword"; //NOSONAR
        name = "None";
        documentType = DocumentType.NONE.getCode();
        documentNum = "0123456789";
        email = "0123456789";
    }

}
