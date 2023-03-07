package preserveOther.entity;

import lombok.Data;

import javax.validation.Valid;

/**
 * @author fdse
 */
@Data
public class TrainType {
    @Valid
    private String id;

    @Valid
    private int economyClass;

    @Valid
    private int confortClass;

    public TrainType(){
        //Default Constructor
    }

    public TrainType(String id, int economyClass, int confortClass) {
        this.id = id;
        this.economyClass = economyClass;
        this.confortClass = confortClass;
    }

}
