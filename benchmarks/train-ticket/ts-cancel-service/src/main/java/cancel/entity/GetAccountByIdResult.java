package cancel.entity;

import lombok.Data;

/**
 * @author fdse
 */
@Data
public class GetAccountByIdResult {

    private boolean status;

    private String message;

    private Account account;

    public GetAccountByIdResult() {
        //Default Constructor
    }

}
