package order.entity;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.UUID;

/**
 * @author fdse
 */
@Data
@AllArgsConstructor
public class OrderAlterInfo {

    private UUID accountId;

    private UUID previousOrderId;

    private String loginToken;

    private Order newOrderInfo;

    public OrderAlterInfo(){
        newOrderInfo = new Order();
    }
}
