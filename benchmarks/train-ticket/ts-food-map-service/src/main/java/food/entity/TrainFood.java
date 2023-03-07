package food.entity;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import javax.validation.constraints.NotNull;
import java.util.List;
import java.util.UUID;

@Data
@Document(collection = "trainfoods")
@JsonIgnoreProperties(ignoreUnknown = true)
public class TrainFood {

    @Id
    private UUID id;

    @NotNull
    private String tripId;

    private List<Food> foodList;

    public TrainFood(){
        //Default Constructor
        this.tripId = "";
    }

}
